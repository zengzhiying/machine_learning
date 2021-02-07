#!/usr/bin/python
# coding=utf-8
"""使用dlib svm分类器训练对象检测器
训练方法: 在程序内部加载图片区域进行训练
"""
import os
import sys
import glob

import dlib
from skimage import io


# 初始化模型
options = dlib.simple_object_detector_training_options()
# 使用对称检测器(比如人脸是左右对称的)
options.add_left_right_image_flips = True
# 分类错误惩罚权重参数C
# 注意: C过大会产生对错误惩罚会很大, 所以容易过拟合, C过小容易出现高偏差, 实际使用中要多尝试几个值并使用验证集选择效果最好的C
options.C = 5
# 使用多少CPU核心并行训练
options.num_threads = 4
options.be_verbose = True

# 训练: 保证输入图像和label列表一致
image_floder = './stools/'
train_image_files = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg']
train_images = []
test_image_files = ['13.jpg', '14.jpg', '15.jpg']
test_images = []
for train_image_file in train_image_files:
    train_images.append(io.imread(image_floder + train_image_file))
for test_image_file in test_image_files:
    test_images.append(io.imread(image_floder + test_image_file))

train_boxes = [([dlib.rectangle(left=371, top=455, right=723, bottom=943)]),
               ([dlib.rectangle(left=357, top=471, right=357+369, bottom=471+555)]),
               ([dlib.rectangle(left=413, top=313, right=413+319, bottom=313+451)]),
               ([dlib.rectangle(left=411, top=399, right=411+351, bottom=399+507)]),
               ([dlib.rectangle(left=287, top=439, right=287+423, bottom=439+537)]),
               ([dlib.rectangle(left=353, top=427, right=353+307, bottom=427+477)]),
               ([dlib.rectangle(left=369, top=569, right=369+385, bottom=569+601)]),
               ([dlib.rectangle(left=401, top=449, right=401+301, bottom=449+451)]),
               ([dlib.rectangle(left=265, top=577, right=265+417, bottom=577+597)]),
               ([dlib.rectangle(left=335, top=311, right=335+397, bottom=311+617)]),
               ([dlib.rectangle(left=341, top=381, right=341+447, bottom=381+645)]),
               ([dlib.rectangle(left=217, top=533, right=217+449, bottom=533+653)])]
test_boxes = [([dlib.rectangle(left=367, top=427, right=367+305, bottom=427+459)]),
              ([dlib.rectangle(left=481, top=309, right=481+473, bottom=309+709)]),
              ([dlib.rectangle(left=261, top=393, right=261+537, bottom=393+777)])]


detector = dlib.train_simple_object_detector(train_images, train_boxes, options)
# 保存模型
detector.save('stools.model')

print("\nTraining accuracy: {}".format(
    dlib.test_simple_object_detector(train_images, train_boxes, detector)))
print(dlib.test_simple_object_detector(test_images, test_boxes, detector))
