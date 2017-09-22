#!/usr/bin/env python
# coding=utf-8

"""传入测试句子 预测模型
"""

import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle

import jieba
import numpy
from sklearn.externals import joblib
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
    # 加载文本对应向量字典
    with open('word_vectors.dict', 'rb') as fp:
        word_vectors = pickle.load(fp)

    # 载入模型
    mlp_model = joblib.load('emotional_all.model')

    test_sentence = "N多年前在上海看《琥珀》，不知道它是廖一梅的，因为有影帝（刘烨）和美女盖住了她的光芒。去年在话剧中心看《恋爱的犀牛》，被一群黄牛围堵，才知道原来她的话剧那么红。这是一本不错的书，记得书中的很多细节，比方说关于男人变老和变成熟，比方说关于美感实不实用，比方说引用的《两只狗的生活意见》里关于门的台词。"
    x = words_to_vector(sentence_to_words(test_sentence))
    predict_result = mlp_model.predict([x])
    if predict_result[0] == 1:
        print("积极.")
    else:
        print("消极.")
