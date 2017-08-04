#!/usr/bin/env python
# coding=utf-8
''' 解析mnist标签样本集标注内容 '''
import struct

mnist_label_file = 'train-labels-idx1-ubyte'  # 二进制图片文件名 解压之后的

# 加载mnist标签数据到内存
fp = open(mnist_label_file, 'rb')
mnist_label = fp.read()
fp.close()

index = 0
# 解析8字节文件头 label_number 标签总个数
magic_number, label_number = struct.unpack_from('>II', mnist_label, index)
index += struct.calcsize('>II')

labels = []

for i in range(label_number):
    # 解析标签值
    label = struct.unpack_from('>B', mnist_label, index)[0]
    print label
    labels.append(str(int(label)))
    index += struct.calcsize('>B')

# 保存标签结果到文件
fp = open('label.txt', 'wb')
fp.write(', '.join(labels))
fp.write('\n')
fp.close()
