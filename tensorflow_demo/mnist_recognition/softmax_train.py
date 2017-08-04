#!/usr/bin/env python
# coding=utf-8
''' 使用softmax回归训练mnist样本
    相当于单层前向传播神经网络
'''
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

# 加载mnist样本 这样会将样本分为6000个训练集和10000个测试集
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
# mnist.train 训练集  mnist.test 测试集
# mnist.train.images 图片集[60000, 784]的张量  mnist.train.label 标签集 [60000, 10]的数字矩阵

# 设置样本占位符 样本行数不固定 列数(features)为28*28 所有样本为一个二维浮点张量
x = tf.placeholder("float", [None, 784])

# 设置权重 并初始化为0
W = tf.Variable(tf.zeros([784, 10]))

# 设置偏置单元
b = tf.Variable(tf.zeros(10))

# 建立模型 参数为x*θ + b这样的线性模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义训练模型 y_ 为样本真实值
y_ = tf.placeholder("float", [None, 10])

# 定义代价函数
cost_function = -1 * tf.reduce_sum(y_ * tf.log(y))
# 使用梯度下降最小化代价函数的值 学习速率0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost_function)

# 初始化变量
init = tf.global_variables_initializer()

# 变量保存初始化
saver = tf.train.Saver()

# 启动模型 开始训练
with tf.Session() as sess:
    sess.run(init)
    # 迭代10000次模型
    train_number = 0
    for i in range(5000):
        # 每次随机抓取100个数据样本
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 运行train_step
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        train_number += 1
        if train_number % 100 == 0:
            print "训练次数: %d" % train_number
    # 训练完毕
    # 评估模型的准确度 argmax返回值为1的索引位置 正好是对应的数字 equal判断是否相等
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 预测正确率
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    # 保存模型
    saver.save(sess, "mnist_model/mnist.model")

