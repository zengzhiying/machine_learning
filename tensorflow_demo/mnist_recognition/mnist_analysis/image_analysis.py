#!/usr/bin/env python
# coding=utf-8
''' 解析mnist图片样本集为图片 '''
from PIL import Image
import struct

mnist_image_file = 'train-images-idx3-ubyte'  # 二进制图片文件名 解压之后的
# 加载mnist数据到内存
fp = open(mnist_image_file, 'rb')
mnist_image = fp.read()
fp.close()

index = 0
# 解析16字节文件头 按文件顺序(大端序)解析 image_number:图片个数  rows*columns:图片分辨率
magic_number, image_number, rows, columns = struct.unpack_from('>IIII', mnist_image, index)
# 移动下标
index += struct.calcsize('>IIII')

# 解析图片数据
for i in range(image_number):
    # 创建空白rows*columns灰度图像
    image = Image.new('L', (rows, columns))
    # 绘制手写字体
    for x in range(rows):
        for y in range(columns):
            # 每次读一个字节写到图像对应的像素点
            image.putpixel((y, x), int(struct.unpack_from('>B', mnist_image, index)[0]))
            index += struct.calcsize('>B')
    # 处理完一张图片
    print "保存图片： %d.png" % i
    image.save('mnist_images/%d.png' % i)
    if i == 100:
        break

