# -*- coding: utf-8 -*-
"""
@Time    : 2025/1/3 下午7:35
@Author  : Kend
@FileName: tools.py
@Software: PyCharm
@modifier:
"""

import cv2


def get_images_by_opencv(video_url):
    """
    获取视频的每一帧图片
    :param video_url:
    :return:
    """
    video_capture = cv2.VideoCapture(video_url)
    count = 0
    while True:
        ret, frame = video_capture.read()
        if ret:
            count += 1
            if count % 10 == 0:
                print(f"已处理{count}帧")
                cv2.imwrite(rf"D:\kend\datesets\frame_{count}.jpg", frame)
        else:
            break

    print("裁剪完成")

if __name__ == '__main__':
    get_images_by_opencv(r"D:\kend\myPython\yolov5-evo\camera001.mp4")