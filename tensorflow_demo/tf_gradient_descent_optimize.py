#!/usr/bin/env python
# coding=utf-8
"""使用tensorflow梯度下降算法对函数优化
得到最小化函数的参数值
"""
import numpy as np
import tensorflow as tf

# 初始值
w = tf.Variable(0, dtype=tf.float32)
# cost function: y = w^2 - 5*w + 10
cost = tf.add(tf.add(w**2, tf.multiply(-5., w)), 10)
# 学习速率: 0.01
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(w))
    for i in range(1000):
        session.run(train)
    print(session.run(w))

