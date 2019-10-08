#!/usr/bin/env python
# coding=utf-8

"""对文本进行简单的情感分类训练
4. 随机打乱数据集并按照80%和20%划分训练集和测试集
5. 输入分类器训练
6. 在测试集上验证效果
由于字典的无序性 多次运行的输入顺序会不一样 但是在单词运行生命周期内的顺序是不变的 所以不影响结果
"""

import sys
import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import jieba
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.decomposition import PCA

reload(sys)
sys.setdefaultencoding('utf-8')

# 设置全局停用词字典
stop_words = {}

def load_stop_words(filename):
    """加载停用词
    将停用词写入stop_words字典中
    """
    with open(filename, 'rb') as fp:
        for line in fp:
            stop_words[line.strip()] = 1

if __name__ == '__main__':
    # 加载停用词
    stop_word_file = './stop_words.txt'
    load_stop_words(stop_word_file)
    # 加载样本
    with open('dataset.mat', 'rb') as fp:
        X = pickle.load(fp)
    with open('label.mat', 'rb') as fp:
        y = pickle.load(fp)
    print("X: %d, y: %d" % (len(X), len(y)))
    time.sleep(3)
    X = numpy.array(X)
    y = numpy.array(y)
    # 随机打乱样本集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # 训练模型 梯度迭代: lbgfs 学习速率: 0.001 隐藏层:2层
    # 注: 样本数量较少时(几百或几千)使用lbgfs拟牛顿法迭代,收敛快;adam可能不收敛
    mlp_model = MLPClassifier(solver='lbgfs', alpha=1e-3,
                              hidden_layer_sizes=(5, 2), random_state=1)
    mlp_model.fit(X_train, y_train)
    print(mlp_model.loss_)
    accuracy = mlp_model.score(X_test, y_test)
    print("accuracy: %g" % accuracy)
    if accuracy > 0.8:
        joblib.dump(mlp_model, 'emotional_reduce.model')


