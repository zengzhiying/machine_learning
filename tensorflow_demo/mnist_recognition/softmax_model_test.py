#!/usr/bin/env python
# coding=utf-8
''' 
    测试训练模型的识别结果
'''
import tensorflow as tf
from PIL import Image

# 加载图片 转换为灰度图像
test_image = Image.open('test.png').convert('L')

# resize调整大小
if test_image.size[0] != 28 or test_image.size[1] != 28:
    test_image = test_image.resize((28, 28))

# 转换图像为28*28维向量
image_vector = []

for i in range(28):
    for j in range(28):
        pixel = float(test_image.getpixel((j, i)))/255.0
        # print pixel
        image_vector.append(pixel)

# 图片向量可直接作为输入 无需转换
# image_vector = tf.Variable(image_vector)

# 构建和训练时一致的模型 接收模型恢复
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))
y = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'mnist_model/mnist.model')
    print image_vector
    # 获取结果集中最大的值对应的行
    predict = tf.argmax(y, 1)
    # 输入x为样本矩阵 [image_vector]
    result = sess.run(predict, feed_dict={x: [image_vector]})[0]
    print "识别结果: %d" % result
