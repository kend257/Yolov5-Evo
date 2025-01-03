# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C2f,
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    """
    定义了一个名为 Detect 的类，它是 YOLOv5 模型中的检测头（detection head），用于处理输入张量并生成目标检测模型的检测结果。
    这个类是 YOLOv5 模型的重要组成部分，负责将模型的中间特征转换为最终的检测结果。
    它通过卷积层输出每个网格点的预测值，然后根据这些预测值和预定义的锚点框生成物体的边界框和类别概率。
    """

    stride = None  # 表示每个检测层的步幅，这在构建过程中计算得出。
    dynamic = False  # 是否强制重建网格（grid）。
    export = False  # 是否处于导出模式。

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # 类别数量，默认为80。
        self.no = nc + 5  # 每个锚点的输出数量，等于类别数量加5（4个坐标值和1个置信度）。
        self.nl = len(anchors)  # 检测层的数量。
        self.na = len(anchors[0]) // 2  # 每个检测层的锚点数量。
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化为空的网格张量列表。
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化为空的锚点网格张量列表。
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 创建一个卷积层的模块列表，每个卷积层输出对应数量的锚点和类别。
        self.inplace = inplace  # 是否使用原地操作。

    # 前向传播函数 forward
    def forward(self, x):
        """
        x: 输入张量，形状为 (bs, 3, ny, nx, 85)，其中 bs 是批量大小，ny 和 nx 是网格的尺寸，
        85 是每个网格点的输出通道数（包括坐标、宽高、置信度和类别概率）。
        """
        z = []  # inference output
        # 遍历每个检测层，应用卷积操作。
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    # 辅助函数 _make_grid
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """
        nx, ny: 网格的宽度和高度。
        i: 当前检测层的索引。
        torch_1_10: 用于检查 PyTorch 版本是否大于等于 1.10.0。
        作用： 该函数生成锚点框的网格和锚点网格，用于将预测结果映射回原始图像坐标。
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    """
    这段代码定义了一个名为 Segment 的类，它继承自 Detect 类，是 YOLOv5 模型用于分割任务的头部（head）。
    Segment 类在此基础上扩展了分割功能，增加了掩码（mask）和原型（prototype）层
    作用： 通过这个 Segment 类，YOLOv5 模型可以同时进行目标检测和分割任务，输出不仅包括目标的边界框和类别，还包括目标的掩码信息。
    这对于需要精确分割目标的应用场景非常有用，例如医学图像分析、自动驾驶等。
    """

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """
        初始化 YOLOv5 分割头，参数包括类别数 (nc)、锚点 (anchors)、掩码数量 (nm)、原型数量 (npr)、输入通道 (ch) 和是否原地操作 (inplace)。
        """
        super().__init__(nc, anchors, ch, inplace)  # 调用父类 Detect 的初始化方法。
        self.nm = nm  # 掩码数量
        self.npr = npr  # 原型数量
        self.no = 5 + nc + self.nm  # 每个锚点的输出数量 (self.no)，包括目标检测的 5 个坐标（中心点坐标、宽高、置信度）和类别数，再加上掩码数量。
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 创建一个卷积层列表 (self.m)，每个卷积层将输入通道转换为每个锚点的输出数量。
        self.proto = Proto(ch[0], self.npr, self.nm)  # 创建一个原型层 (self.proto)，用于生成掩码原型。
        self.detect = Detect.forward

    def forward(self, x):
        """

        """
        p = self.proto(x[0])  # Proto类 ： 用于生成掩码原型。它接收输入通道、原型数量和掩码数量作为参数。
        # 处理输入，通过原型层生成掩码原型 (p)。
        x = self.detect(self, x)  # 调用 Detect 类的前向传播方法 (self.detect) 生成检测结果 (x)。这里将其赋值给 self.detect 以便在 forward 方法中调用。
        # 根据模型是否在训练模式 (self.training) 或导出模式 (self.export)，调整返回的输出：
        ''' 
        训练模式：返回检测结果和掩码原型。
        导出模式：返回检测结果的第一部分和掩码原型。
        其他情况：返回检测结果的第一部分、掩码原型和检测结果的其他部分。
        '''
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])



class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(self, x, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self



"""
DetectionModel 类是基于 YOLOv5 架构设计的一个目标检测模型，
用于执行对象检测任务。这个类支持自定义配置、输入通道数、类别数量和锚点设置。
"""
class DetectionModel(BaseModel):
    """"
    初始化过程中，首先读取配置文件并将其存储在 self.yaml 中。
    然后根据提供的参数调整配置文件中的相应值。接下来，调用 parse_model 函数根据配置构建模型，
    并初始化一些额外的属性，如类别名称列表 self.names 和是否启用原地操作的标志 self.inplace。
    """
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # cfg: 配置文件路径或字典，包含模型结构和其他必要的设置。
        else:  # 否则，假设 cfg 是一个 YAML 文件路径，读取文件内容并解析为字典。
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # ch: 输入图像的通道数，默认为3（即RGB图像）。设置输入通道数 ch，优先使用配置文件中的值，如果没有则使用传入的值。
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # nc: 模型要识别的类别数量，如果提供了类别数量 nc 并且与配置文件中的值不同，则覆盖配置文件中的值，
        if anchors: # anchors: 自定义锚点列表，如果提供，将覆盖配置文件中的默认锚点。
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        # 使用 parse_model 函数根据配置文件解析模型结构
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # 返回模型和保存列表。
        self.names = [str(i) for i in range(self.yaml["nc"])]  # 初始化类别名称列表 self.names，默认为 [0, 1, 2, ..., nc-1]。
        self.inplace = self.yaml.get("inplace", True)  # 设置是否启用原地操作的标志 self.inplace。

        # 构建步长和锚点
        m = self.model[-1]  # 如果模型的最后一层是 Detect 或 Segment 类型，则定义一个内部前向传播函数 _forward。
        if isinstance(m, (Detect, Segment)):

            def _forward(x):
                # 则定义一个内部前向传播函数 _forward。
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            s = 256  # 设置最小步长 s 为 256。
            m.inplace = self.inplace  # 设置 m.inplace 为 self.inplace。
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # 计算模型的输出步长 m.stride。
            check_anchor_order(m)  # 检查锚点顺序。
            m.anchors /= m.stride.view(-1, 1, 1)  # 根据步长调整锚点大小。
            self.stride = m.stride   # 初始化步长 self.stride。
            self._initialize_biases()  # 调用 _initialize_biases 方法初始化偏置。

        # 初始化模型的所有权重和偏置。
        initialize_weights(self)
        # 打印模型信息。
        self.info()
        # 记录日志。
        LOGGER.info("")

    # 前向传播方法 forward
    def forward(self, x, augment=False, profile=False, visualize=False):
        """如果启用增强推理 augment，调用 _forward_augment 方法。"""
        if augment:
            return self._forward_augment(x)
        # 否则，调用 _forward_once 方法进行单尺度推理。
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # 增强推理方法 _forward_augment
    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # 获取输入图像的尺寸 img_size。
        s = [1, 0.83, 0.67]  # 定义缩放因子 s 和翻转类型 f。
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # 初始化输出列表 y。
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(
                self.stride.max()))  # 对每个缩放因子和翻转类型组合，生成新的输入图像 xi。
            yi = self._forward_once(xi)[0]  # 进行前向传播，获取输出 yi。
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)   # 反缩放和反翻转预测结果 yi。
            y.append(yi)  # # 将处理后的结果添加到输出列表 y 中。
        y = self._clip_augmented(y) # 裁剪增强推理的结果。
        return torch.cat(y, 1), None  # 将所有结果拼接在一起并返回。

    # 反缩放和反翻转方法 _descale_pred
    def _descale_pred(self, p, flips, scale, img_size):
        """如果启用原地操作 self.inplace，直接在 p上进行反缩放和反翻转。"""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
        # 否则，创建新的张量进行反缩放和反翻转，然后重新拼接。
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p


    def _clip_augmented(self, y):
        """
        裁剪增强推理结果方法 _clip_augmented
        """
        nl = self.model[-1].nl  # 获取检测层数 nl。
        g = sum(4**x for x in range(nl))  # 计算网格点数 g。
        e = 1  # 设置排除的层数 e。
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # 计算需要裁剪的索引 i。
        y[0] = y[0][:, :-i]  # 放大
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # 缩小
        return y

    # 初始化偏置方法 _initialize_biases
    def _initialize_biases(self, cf=None):
        """
        初始化模型的所有权重和偏置方法 _initialize_biases
        """
        m = self.model[-1]  # 获取模型的最后一层 m。
        for mi, s in zip(m.m, m.stride):  # 遍历每个检测层 mi 和对应的步长 s。
            b = mi.bias.view(m.na, -1)  # 将偏置 b 从 (255,) 形状转换为 (3, 85) 形状。
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # 初始化对象检测偏置 b.data[:, 4]。
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # 初始化类别检测偏置 b.data[:, 5 : 5 + m.nc]。
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)   # 更新偏置参数 mi.bias。


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    """YOLOv5 segmentation model for object detection and segmentation tasks with configurable parameters."""

    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list)."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """YOLOv5 classification model for image classification tasks, initialized with a config file or detection model."""

    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None



# 解析模型结构
"""
parse_model 动态解析 YOLOv5 的配置，生成一个由 PyTorch 构建的模型结构，同时保留需要的中间输出，用于特定任务（如目标检测）。
它通过多种模块的特殊处理，确保生成的模型高效且灵活。
"""
def parse_model(d, ch):
    """
    它用于从字典 d 中解析 YOLOv5 模型，并根据输入通道 ch 和模型架构配置模型层。
    :param d: 包含模型配置的字典d：一个包含模型配置的字典，定义了模型的结构、参数等。
c   :param h：一个列表，表示每一层的输入通道数。
    return: 返回一个包含模型层的列表。动态创建并返回一个可用的模型层（PyTorch 的 nn.Sequential）
    """
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # note 从字典 d 中提取模型配置参数，包括锚点 (anchors)、类别数 (nc)、深度倍增 (gd)、宽度倍增 (gw)、激活函数 (act) 和通道倍增 (ch_mul)。
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],  # 定义锚框形状
        d["nc"],    # 目标检测的类别数量。
        d["depth_multiple"],    # 用于调整模型深度。
        d["width_multiple"],    # 用于调整模型宽度。
        d.get("activation"),    # 定义使用哪种激活函数（默认为 SiLU，如果提供了，重新定义
        d.get("channel_multiple"), # 通道倍增 (ch_mul): 如果未提供，默认为 8。
    )
    # 如果定义了激活函数 (act)，则重新定义默认激活函数。
    """
    这行代码的目的是重新定义默认的激活函数（activation function）为 SiLU（Sigmoid Linear Unit）激活函数。具体来说，它将 Conv 类的 default_act 属性设置为 nn.SiLU()。
    1、实现原理
    Conv 类：假设 Conv 是一个定义了卷积操作的类，通常用于深度学习模型中。这个类可能包含一些默认的参数或属性，比如默认的激活函数。
    default_act 属性：这是 Conv 类中的一个属性，用于存储默认的激活函数。激活函数在神经网络中用于引入非线性，使得模型能够学习更复杂的特征。
    nn.SiLU()：这是 PyTorch 中的一个激活函数，表示 Sigmoid Linear Unit。SiLU 是一种相对较新的激活函数，公式为 SiLU(x) = x * sigmoid(x)，其中 sigmoid(x) 是 sigmoid 函数。
    2、重新定义默认激活函数的用途包括：
    简化模型定义：在创建卷积层时，不需要每次都显式地指定激活函数，而是使用默认的 SiLU 激活函数。
    一致性：确保模型中的所有卷积层都使用相同的激活函数，避免因激活函数不同而导致的模型性能差异。
    """
    if act:
        Conv.default_act = eval(act)  # # 动态执行字符串，设置默认激活函数
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    # 如果没有定义通道倍增 (ch_mul)，则默认设为 8。
    if not ch_mul:
        ch_mul = 8
    # 计算锚点的数量 (na) 和每个锚点的输出数量 (no)。
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  #
    # 锚点数量
    no = na * (nc + 5)  # 锚点数量*(类别+5)

    """
    layers: 保存模型层的列表。
    save: 保存需要在前向传播中保留的层索引。
    c2: 当前层的输出通道数，初始为 ch[-1]（最后一层的通道数）。
    """
    layers, save, c2 = [], [], ch[-1]  # 初始化层和保存列表
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # 模型的架构配置由 backbone 和 head 组成
        # f: 输入层索引。
        # n: 重复次数（乘以 gd 调整深度）。
        # m: 模块类型（如 Conv、C3 等）。
        # args: 模块的参数
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 模块处理： 不同类型的模块需要特殊处理：
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C2f,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]    # 输入通道数、目标输出通道数
            if c2 != no:  # # 如果不是输出层
                c2 = make_divisible(c2 * gw, ch_mul) #  # 调整通道数，确保能被 ch_mul 整除

            args = [c1, c2, *args[1:]]   # # 参数更新
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x, C2f}:
                args.insert(2, n)  # 添加重复次数
                n = 1   # # 重复次数设置为1


        elif m is nn.BatchNorm2d:
            args = [ch[f]]

        # 目的：计算拼接后通道数。
        elif m is Concat:
            c2 = sum(ch[x] for x in f)  # # 拼接通道数
        # TODO: channel, gw, gd

        # Detect/Segment模块  Detect/Segment 用于目标检测和分割，需根据输入调整参数。
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])  # # 输入通道列表
            if isinstance(args[1], int):  # # 如果锚点数是整数
                args[1] = [list(range(args[1] * 2))] * len(f)   # # 默认生成锚点
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)   # # 调整通道数
        # 特殊模块（Contract/Expand等）
        elif m is Contract:  # 将特征图压缩。
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:   # 将特征图扩展。
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]  #
        # 构造模块  如果 n > 1，将模块重复 n 次。
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 保存层信息
        t = str(m)[8:-2].replace("__main__.", "")  # # 模块类型
        np = sum(x.numel() for x in m_.parameters())  # # 参数总数
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # # 保存层信息
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # # 打印层信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 将需要保存的层索引添加到 save。
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)  # 更新 ch，为下一层提供输入通道数。
    # save 包含需要保留的层索引，用于前向传播时的中间输出。
    return nn.Sequential(*layers), sorted(save)   # 模型层序列：nn.Sequential(*layers) 是一个完整的模型结构。




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # 创建模型
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # 优化器
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
