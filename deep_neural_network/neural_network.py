# coding=utf-8

import numpy as np

class NeuralNetwork(object):
    """简单的深度神经网络构建类
    实现一个三层的神经网络(两层隐藏层)
    """
    def __init__(self):
        # 学习速率 (多次调试选择使得损失函数下降最好的值)
        self.epsilon = 0.0155
        # 隐藏层个数
        self.hidden_layer_number = 2
        # 隐藏层单元个数
        self.hidden_number = 4
        # 网络总层数
        self.L = 3
        # 初始化参数缩放倍数
        self.init_zoom = 0.01

    def train(self, X_train, Y_train, iterations_number=1000):
        """训练神经网络
        Args:
            X_train: 训练样本集 n*m
            Y_train: 训练label 1*m
            iterations_number: 迭代次数 默认:1000
        Returns:
            直接更新网络参数 无返回值
        """
        # 随机初始化参数
        self.W1 = np.random.randn(self.hidden_number, np.size(X_train, 0)) * self.init_zoom
        self.b1 = np.zeros((self.hidden_number, 1))
        self.W2 = np.random.randn(self.hidden_number, self.hidden_number) * self.init_zoom
        self.b2 = np.zeros((self.hidden_number, 1))
        self.W3 = np.random.randn(1, self.hidden_number) * self.init_zoom
        self.b3 = np.zeros((1, 1))

        # 样本个数m
        m = X_train.shape[1]

        # 批量梯度下降
        for i in xrange(iterations_number):
            # 前向传播
            Z1 = self.W1.dot(X_train) + self.b1   # 4*n n*m => 4*m
            A1 = self.relu(Z1)
            Z2 = self.W2.dot(A1) + self.b2   # 4*4 4*m => 4*m
            A2 = self.relu(Z2)
            Z3 = self.W3.dot(A2) + self.b3   # 1*4 4*m => 1*m
            A3 = self.sigmod(Z3)

            # 反向传播
            dZ3 = A3 - Y_train
            dW3 = 1.0/m * np.dot(dZ3, np.transpose(A2))   # 1*m m*4 => 1*4
            db3 = 1.0/m * np.sum(dZ3, axis=1, keepdims=True)   # 1*m => 1*1
            dA2 = np.dot(np.transpose(self.W3), dZ3)   # 4*1 1*m => 4*m

            dZ2 = dA2 * self.relu_derive(Z2)   # 4*m  4*m  => 4*m
            dW2 = 1.0/m * np.dot(dZ2, np.transpose(A1))   # 4*m  m*4 => 4*4
            db2 = 1.0/m * np.sum(dZ2, axis=1, keepdims=True)   # 4*m => 4*1
            dA1 = np.dot(np.transpose(self.W2), dZ2)   # 4*4  4*m  =>  4*m

            dZ1 = dA1 * self.relu_derive(Z1)   # 4*m  4*m  => 4*m
            dW1 = 1.0/m * np.dot(dZ1, np.transpose(X_train))   # 4*m  m*n  =>  4*n
            db1 = 1.0/m * np.sum(dZ1, axis=1, keepdims=True)   # 4*n  =>   4*1

            # 梯度下降更新参数
            self.W1 -= self.epsilon * dW1
            self.b1 -= self.epsilon * db1
            self.W2 -= self.epsilon * dW2
            self.b2 -= self.epsilon * db2
            self.W3 -= self.epsilon * dW3
            self.b3 -= self.epsilon * db3

            if i % 10 == 0:
                print("迭代次数: %d loss: %g" % ((i + 1), self.loss_calculate(X_train, Y_train)))


    def loss_calculate(self, X, Y):
        """损失函数计算"""
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.relu(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        A3 = self.sigmod(Z3)

        m = Y.shape[1]
        loss = -1.0/m * np.sum(Y * np.log(A3) + (1 - Y) * np.log(1 - A3))
        return loss

    def predict(self, x):
        """对输入样本进行预测"""
        z1 = self.W1.dot(x) + self.b1   # 4*n  n*1  => 4*1
        a1 = self.relu(z1)
        z2 = self.W2.dot(a1) + self.b2   # 4*4  4*1 => 4*1
        a2 = self.relu(z2)
        z3 = self.W3.dot(a2) + self.b3   # 1*4  4*1 => 1*1
        a3 = self.sigmod(z3)

        return a3


    def relu(self, x):
        """激活函数: ReLU
        修正线性单元
        """
        return np.maximum(0, x)

    def sigmod(self, x):
        """输出层激活函数: sigmod"""
        return 1 / (1 + np.exp(-x))

    def relu_derive(self, x):
        """ReLU函数导数"""
        x_copy = np.array(x)
        x_copy[x_copy >= 0] = 1
        x_copy[x_copy < 0] = 0
        return x_copy
