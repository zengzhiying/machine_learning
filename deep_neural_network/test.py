#!/usr/bin/env python
# coding=utf-8
"""调用神经网络模型
对样本做简单的训练和学习, 实现估计一个矩形的面积是否大于某个值
"""
import numpy as np

from neural_network import NeuralNetwork

if __name__ == '__main__':
    model = NeuralNetwork()
    X = []
    Y = []
    X_test = []
    Y_test = []
    train_num = 1000
    test_num = 20
    for i in range(train_num):
        a = np.random.randint(1, 201)
        b = np.random.randint(1, 201)
        s = a * b
        X.append([a, b])
        if s > 6000:
            Y.append(1)
        else:
            Y.append(0)

    X = np.transpose(np.array(X))
    Y = np.array(Y).reshape(1, train_num)
    print(X)
    print(Y)

    for i in range(test_num):
        a = np.random.randint(1, 201)
        b = np.random.randint(1, 201)
        s = a * b
        X_test.append([a, b])
        if s > 6000:
            Y_test.append(1)
        else:
            Y_test.append(0)

    X_test = np.transpose(np.array(X_test))
    Y_test = np.array(Y_test).reshape(1, test_num)
    print(X_test)
    print(Y_test)

    model.train(X, Y, iterations_number=20000)
    p = model.predict(X_test)
    print(p)

    p[p >= 0.5] = 1
    p[p < 0.5] = 0
    print(p == Y_test)
    print("准确率: %g" % (np.sum(p == Y_test)*1.0/Y_test.shape[1]))

    print(model.predict(np.array([[28],[32]])))
    print(model.predict(np.array([[163], [254]])))
