#!/usr/bin/python
"""使用dlib内置的人脸检测器检测人脸""" 

import sys

import dlib
import numpy as np
from skimage import io, draw


detector = dlib.get_frontal_face_detector()

for f in sys.argv[1:]:
    print("Processing file: {}".format(f))
    img = io.imread(f)
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, detect in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, detect.left(), detect.top(), detect.right(), detect.bottom()))
        Y = np.array([detect.top(), detect.top(), detect.bottom(), detect.bottom()])
        X = np.array([detect.left(), detect.right(), detect.right(), detect.left()])
        rr, cc = draw.polygon_perimeter(Y, X)
        draw.set_color(img, [rr, cc], [255, 0, 0])
        io.imsave('detector.jpg', img)
    dlib.hit_enter_to_continue()


# 检测人脸得分和识别不同方向的人脸
if (len(sys.argv[1:]) > 0):
    img = io.imread(sys.argv[1])
    dets, scores, idx = detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))

