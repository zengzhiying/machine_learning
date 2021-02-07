#!/usr/bin/env python3
# coding=utf-8
"""随机分割当前目录下所有图片数据集为训练集和测试集
"""

import os
import random

train_floder = './train'
validation_floder = './cv'
train_ratio = 0.8
dataset_floder = './dataset'
# validation_ratio = 0.2

if __name__ == '__main__':
    dataset = []
    for parent, dirnames, filenames in os.walk(dataset_floder):
        for filename in filenames:
            print(filename)
            dataset.append(filename)

    dataset_number = len(dataset)
    print("样本总个数: %d" % dataset_number)
    print("开始随机分割.")
    validation_number = int(dataset_number * 0.2)
    train_number = dataset_number - validation_number
    print("训练集数量: %d 测试集数量: %d" % (train_number, validation_number))

    # 随机打乱样本
    for i in range(dataset_number):
        index = random.randint(0, dataset_number - 1)
        dataset.append(dataset.pop(index))
    # 移动训练集合测试集
    for i, image_file in enumerate(dataset):
        if i < train_number:
            os.system('mv {} {}'.format(image_file, train_floder))
        else:
            os.system('mv {} {}'.format(image_file, validation_floder))

