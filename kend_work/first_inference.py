# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/19 下午9:06
@Author  : Kend
@FileName: first_inference.py
@Software: PyCharm
@modifier:
"""

import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5n")  # or yolov5n -
# print(model)

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()