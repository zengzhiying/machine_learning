#!/usr/bin/env python
# coding=utf-8

"""初始加载数据集并序列化
以供后来模型训练和测试使用
1. 加载文本->分词->去除停用词
2. 根据词语出现情况确定输入向量
3. 分别将单个样本转换为向量并对应标签向量 形成数据集
4. 进行维数约减(原数据直接序列化超大,模型训练非常慢,所以此处添加pca操作)
5. 将处理后的数据集序列化到硬盘
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
from sklearn.decomposition import PCA
from sklearn.externals import joblib
reload(sys)
sys.setdefaultencoding('utf-8')

# 设置全局停用词字典
stop_words = {}
# 设置文本对应向量字典和索引对应单词字典
word_vectors = {}
index_words = {}
words = set([])

def load_stop_words(filename):
    """加载停用词
    将停用词写入stop_words字典中
    """
    with open(filename, 'rb') as fp:
        for line in fp:
            stop_words[line.strip()] = 1

def __load_dataset(filename):
    """依次加载样本
    然后将每行样本进行分词并去除停用词
    最终将结果set中
    """
    with open(filename, 'rb') as fp:
        for line in fp:
            seg_list = jieba.cut(line.strip(), cut_all=False)
            for seg in seg_list:
                if seg.strip() == '':
                    continue
                if not seg.encode('utf-8') in stop_words:
                    words.add(seg)

def __dataset_to_vector():
    """用已有的set建立词语向量和词语索引字典
    索引编号从1开始
    """
    for i, w in enumerate(words):
        word_vectors[w] = i + 1
        index_words[i + 1] = w

def words_to_vector(words):
    """分词列表转成向量
    """
    x = numpy.zeros([len(word_vectors)])
    for word in words:
        index = word_vectors[word]
        x[index - 1] = 1
    return x

def sentence_to_words(sentence):
    """对一句话分词并转换为分词列表
    """
    words = []
    participles = jieba.cut(sentence)
    for participle in participles:
        if (participle.strip() != '' and
            participle.encode('utf-8') not in stop_words):
            words.append(participle)
    return words

if __name__ == '__main__':
    # 加载停用词
    stop_word_file = './stop_words.txt'
    load_stop_words(stop_word_file)
    # 加载样本初始化向量字典
    positive_file = './positive.dat'
    negative_file = './negative.dat'
    __load_dataset(positive_file)
    __load_dataset(negative_file)
    __dataset_to_vector()
    print("向量维度: {}".format(len(words)))
    # 加载样本并整理
    X = []
    y = []
    # 加载正样本
    with open(positive_file, 'rb') as fp:
        for line in fp:
            single_x = sentence_to_words(line.strip())
            X.append(words_to_vector(single_x))
            y.append(1)
    # 加载负样本
    with open(negative_file, 'rb') as fp:
        for line in fp:
            single_x = sentence_to_words(line.strip())
            X.append(words_to_vector(single_x))
            y.append(0)
    print("X: %d, y: %d" % (len(X), len(y)))
    time.sleep(3)
    # 分割一小部分数据做降维
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    print(pca.explained_variance_ratio_)
    X = pca.transform(X)
    print(X.shape)
    # 序列化模型和数据
    joblib.dump(pca, 'pca.model')
    with open('word_vectors.dict', 'wb') as fp:
        pickle.dump(word_vectors, fp)
    with open('dataset.mat', 'wb') as fp:
        pickle.dump(X, fp)
    with open('label.mat', 'wb') as fp:
        pickle.dump(y, fp)
