#!/usr/bin/env python3
# coding=utf-8
import pickle

import jieba
import numpy
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

stop_words = {}
words = set([])
word_indexs = {}

def load_stop_words(filename):
    """加载停用词"""
    with open(filename) as f:
        for line in f:
            stop_words[line.strip()] = 1

def content_to_words(content):
    """评价内容分词处理"""
    content_words = []
    participles = jieba.cut(content)
    for participle in participles:
        if participle.strip() in words:
            content_words.append(participle.strip())

    return content_words

def word_to_indexs(content_words):
    """分词列表转换为索引"""
    temp_indexs = []
    for s_word in content_words:
        if s_word in word_indexs:
            temp_indexs.append(word_indexs[s_word])
        else:
            print("ERROR! word: %s 未找到下标！" % s_word)

    return temp_indexs


def cnn_model(words_size, max_features):
    model = Sequential()
    # 此处包含下标0, 因此必须+1
    model.add(Embedding(words_size + 1,
                        16,
                        input_length=max_features))
    model.add(Conv1D(100,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

load_stop_words('stop_words')
with open('words.dat', 'rb') as f:
    words = pickle.load(f)
with open('word_indexs.dat', 'rb') as f:
    word_indexs = pickle.load(f)
with open('max_features', 'r') as f:
    max_features = int(f.read())

model = cnn_model(len(words), max_features)
model.load_weights('hotel_cnn_model.h5')



test_content1 = "此次入住此酒店，感受非常好，尤其前台小周的服务，必须点赞，孩子玩迪士尼中暑，小周安排酒店接送，提前打开房间的空调，并送来果盘，晚上送来洗脚桶泡脚，真的非常温暖，酒店还有机器人服务，可以送餐到房间，免费洗衣烘干，非常现代化，五星级服务，推荐大家入住。"
test_content2 = "不知道酒店的好评是怎样来的，很不满意的一次住宿，不建议一个人住，酒店设施陈旧，乌漆灯黑，就好像很古老很古老的大宅，晚上弱弱的几个黄色壁灯，很像凶宅，鬼屋那样，在体育馆外围，只有一个电梯，从电梯走到房间也要5分钟，远得很，卫生也垃圾得不得了，不会再入住，不建议，不推荐，离地铁口也远，晚上步行出门周边也是很黑很偏僻没什么路灯。早餐没吃，空调声音太大，晚上没有热水洗澡，外卖不能送上楼，没有数据线外借，前台的服务也不行。差"
test_content3 = "我就服了，你们酒店是不是空调就是摆设，两次开的房间空调开了都跟没开似的，你们平时不检修的吗？这大热天三十六七度的住你们家夜里怎么睡？？？"
test_content4 = "真的是市中心的位置了，淮海中路边上，附近有好多美食和商业中心非常方便了，有着loft的楼层格局像是私人阁楼一样，房间虽然不是太大竟然还拥有阳台也是爱了爱了，隔音遮光防潮做的都很不错，非常推荐来这里"
test_words1 = content_to_words(test_content1)
test_words2 = content_to_words(test_content2)
test_words3 = content_to_words(test_content3)
test_words4 = content_to_words(test_content4)
test_words = [test_words1, test_words2, test_words3, test_words4]
for words in test_words:
    if words:
        print(words)
        test_indexs = word_to_indexs(words)
        # print(test_indexs)
        test_indexs = numpy.array([test_indexs])
        test_indexs = sequence.pad_sequences(test_indexs, maxlen=max_features)
        p = model.predict(test_indexs)
        print(p)

