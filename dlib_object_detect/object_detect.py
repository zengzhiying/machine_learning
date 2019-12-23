#!/usr/bin/env python3
# coding=utf-8
"""加载已经训练好的模型, 对图片进行检测
通过命令行传入图片, 打印识别区域结果并对物体框选生成新的图片
"""
import os
import sys

import dlib
import numpy as np
from skimage import io
from skimage import draw

if len(sys.argv) != 2:
    print("请传入图片.")
    sys.exit(1)

object_image_name = sys.argv[1]

# 载入训练好的模型
detector = dlib.simple_object_detector("detector.model")

image = io.imread(object_image_name)
detects = detector(image)
print("识别对象个数: %d" % len(detects))

if detects:
    for k, detect in enumerate(detects):
        print("{} 区域坐标: 左:{}, 上:{}, 右:{}, 下:{}".format(k,
                                                               detect.left(),
                                                               detect.top(),
                                                               detect.right(),
                                                               detect.bottom()))
        # 对图片框选
        Y = np.array([detect.top(), detect.top(), detect.bottom(), detect.bottom()])
        X = np.array([detect.left(), detect.right(), detect.right(), detect.left()])
        rr, cc = draw.polygon_perimeter(Y, X)
        draw.set_color(image, [rr, cc], [255, 0, 0])
        io.imsave('./%s_detect.%s' % (object_image_name.split('.')[0], object_image_name.split('.')[1]), image)
