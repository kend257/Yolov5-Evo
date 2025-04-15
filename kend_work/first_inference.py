# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/19 下午9:06
@Author  : Kend
@FileName: first_inference.py
@Software: PyCharm
@modifier:
"""
import sys
import time
import traceback

import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes


def model_load():
    weights = r"E:\Guanxin_Work\code\work\yolov5\Yolov5-Evo\kend_work\best.pt"
    device = torch.device("cpu")
    model = DetectMultiBackend(weights, device)
    print(f"模型加载完成, device:  {str(device)} ")
    return model


def test_inference_times(model, img):
    try:
        img0 = img.copy()
        # im = letterbox(img, (640, 640), 32, auto=True)[0]  # padded resize
        img_shape_2 = img.shape[:2]
        if True:
            h, w = img_shape_2
            new_h, new_w = 640, 640
            r = min(new_h / h, new_w / w)
            new_unpad = int(round(w * r)), int(round(h * r))
            dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]  # wh padding
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
            dw /= 2  # divide padding into 2 sides
            dh /= 2
            # print('new_unpad:', new_unpad, flush=True)  # (360, 640, 3)
            im = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to("cpu")
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        results = dict()
        # 人体检测 ===================================================================================
        person_pre = model(im)[0]
        person_pre = non_max_suppression(
            person_pre,
            conf_thres=0.4,
            iou_thres=0.45,
            classes=None,
            agnostic=True,
            max_det=100,
            nm=0  # 目标检测设置为0
        )

        person_pre[0][:, :4] = scale_boxes(im.shape[2:], person_pre[0][:, :4], img.shape).round()

    except Exception as e:
        print("区域入侵算法处理异常", e, flush=True)
        error_info = sys.exc_info()
        error_tb = traceback.format_tb(error_info[2])
        for line in error_tb:
            print(line.strip())



def all_time_model(model, img):
    test_inference_times(model, img)  # 除去加载的时间
    beagin_time = time.time()
    for i in range(100):
        test_inference_times(model, img)
    end_time = time.time()
    print(f"100次推理测试，一共耗时：{end_time - beagin_time : .4f} S")
    print(f"100次推理测试，平均一次推理耗时：{(end_time - beagin_time) / 100 * 1000 : .4f} MS")


if __name__ == '__main__':
    # Model
    # model = torch.hub.load("ultralytics/yolov5", "yolov5n")  # or yolov5n -
    model = model_load()
    # Images
    img_path = r"E:\Guanxin_Work\code\work\yolov5\Yolov5-Evo\data\images\zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
    img = cv2.imread(img_path)
    all_time_model(model, img)




