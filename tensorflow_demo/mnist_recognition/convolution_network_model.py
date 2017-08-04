#!/usr/bin/env python
# coding=utf-8
# 使用卷积神经网络实现手写数字识别测试
import tensorflow as tf
from PIL import Image

# 权重正态分布随机初始化 标准差为: 0.1
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 偏置单元权重初始化 全部初始化为0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义卷积函数 步长为1 卷积方式为SAME
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 定义池化函数 卷积是一点点来移动 池化是按照块来不重叠计算 比如20*20 使用10*10的矩阵池化后-> 2*2的
# 池化也是采用2*2大小的模板来执行
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


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

# 定义模型
x = tf.placeholder(tf.float32, [None, 784], name="train_x")
# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 转换输入参数
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 执行第一层卷积和池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# 执行第二层卷积和池化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 全连接层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout层 下面两行在实际应用阶段可以注释掉
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# softmax输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

saver = tf.train.Saver()

# 测试模型
with tf.Session() as sess:
    saver.restore(sess, 'mnist_model/mnist.model')

    predict = tf.argmax(y_conv, 1)

    # 注意这里如果有占位符keep_prob也就是上面两行没有注释 则必须为字典提供值建议:keep_prob: 1
    result = sess.run(predict, feed_dict={x: [image_vector]})
    print result[0]

