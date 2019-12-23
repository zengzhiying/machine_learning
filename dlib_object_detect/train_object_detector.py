#!/usr/bin/python3
# coding=utf-8
"""使用dlib svm分类器训练对象检测器
训练方法: 通过命令行参数传入要训练的图片目录自动开始训练
"""
import os
import sys
import glob

import dlib
from skimage import io

if len(sys.argv) != 2:
    print(
        "Give the path to the examples/faces directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_object_detector.py ../examples/faces")
    exit()
faces_folder = sys.argv[1]


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

# 和上面图片文件对应的训练标签集合测试标签集 (具体训练过程是先通过标签集找到图片然后训练, 所以这里xml其实就是训练集)
training_xml_path = os.path.join(faces_folder, "training.xml")
testing_xml_path = os.path.join(faces_folder, "testing.xml")

# 开始执行训练并保存训练模型
dlib.train_simple_object_detector(training_xml_path, "detector.model", options)

# 打印查看训练集和测试集准确率
print("")
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.model")))

print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(testing_xml_path, "detector.model")))

test_accuracy = dlib.test_simple_object_detector(testing_xml_path, "detector.model")
print(test_accuracy)

dlib.hit_enter_to_continue()

# 不用加载xml标注集也可以使用下面的方式训练
# 保证输入图像和label列表一致即可
images = [io.imread(faces_folder + '/2008_002506.jpg'),
          io.imread(faces_folder + '/2009_004587.jpg')]

boxes_img1 = ([dlib.rectangle(left=329, top=78, right=437, bottom=186),
               dlib.rectangle(left=224, top=95, right=314, bottom=185),
               dlib.rectangle(left=125, top=65, right=214, bottom=155)])
boxes_img2 = ([dlib.rectangle(left=154, top=46, right=228, bottom=121),
               dlib.rectangle(left=266, top=280, right=328, bottom=342)])
boxes = [boxes_img1, boxes_img2]


detector2 = dlib.train_simple_object_detector(images, boxes, options)
# 保存模型
#detector2.save('detector2.svm')

# 打印使用内部训练的准确率
print("\nTraining accuracy: {}".format(
    dlib.test_simple_object_detector(images, boxes, detector2)))
dlib.hit_enter_to_continue()



# 载入训练好的模型
detector = dlib.simple_object_detector("detector.model")

# 显示检测结果
for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    dets = detector(img)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    dlib.hit_enter_to_continue()
