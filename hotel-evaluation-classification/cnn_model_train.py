#!/usr/bin/env python3
# coding=utf-8
import os
import csv

import jieba
import numpy

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

# 词语库
words = set([])
# 停用词典
stop_words = {}
# 词转成下标数字
word_indexs = {}

def load_data(csv_name):
    """加载csv数据文件"""
    dataset_x, dataset_y = [], []
    with open(csv_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if row[0] != 'message':
                dataset_x.append(row[0].strip())
                dataset_y.append(int(row[1]))

    return dataset_x, dataset_y

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


if __name__ == '__main__':
    train_eval, train_y = load_data('hotel_train.csv')
    valid_eval, valid_y = load_data('hotel_test.csv')
    print("train num: %d, valid num: %d" % (len(train_eval), len(valid_eval)))
    load_stop_words("stop_words")
    # 收集词语库
    for content in train_eval:
        segs = jieba.cut(content, cut_all=False)
        for seg in segs:
            # 去掉多余的数字
            if not seg.strip() or seg.isdigit():
                continue
            if not seg.strip() in stop_words:
                words.add(seg)

    for content in valid_eval:
        segs = jieba.cut(content, cut_all=False)
        for seg in segs:
            # 去掉多余的数字
            if not seg.strip() or seg.isdigit():
                continue
            if not seg.strip() in stop_words:
                words.add(seg)

    for i, w in enumerate(words):
        word_indexs[w] = i + 1

    print("words size: %d" % len(words))

    train_words = []
    y = []
    for i, content in enumerate(train_eval):
        content_words = content_to_words(content)
        if content_words:
            train_words.append(content_words)
            y.append(train_y[i])
        else:
            print("content invalid.")

    # print(train_words[:3], y[:3])
    train_indexs = []
    max_features = 0
    for t_words in train_words:
        t_indexs = word_to_indexs(t_words)
        if not t_indexs:
            sys.exit(-1)
        if len(t_indexs) > max_features:
            max_features = len(t_indexs)
        train_indexs.append(t_indexs)

    valid_words = []
    for i, content in enumerate(valid_eval):
        content_words = content_to_words(content)
        if content_words:
            valid_words.append(content_words)
        else:
            print("valid content invalid.")

    valid_indexs = []
    for v_words in valid_words:
        v_indexs = word_to_indexs(v_words)
        if not v_indexs:
            sys.exit(-1)
        if len(v_indexs) > max_features:
            print("valid update max_features: %d", len(v_indexs))
            max_features = len(v_indexs)
        valid_indexs.append(v_indexs)

    # print(train_indexs[:3], y[:3])
    print("max features: %d" % max_features)
    os.system(f'echo {max_features} > max_features')
    train_indexs = numpy.array(train_indexs)
    y = numpy.array(y)
    print("train x shape: {}, y shape: {}".format(train_indexs.shape, y.shape))
    train_indexs = sequence.pad_sequences(train_indexs, maxlen=max_features)

    valid_indexs = numpy.array(valid_indexs)
    valid_y = numpy.array(valid_y)
    print("valid x shape: {}, y shape: {}".format(valid_indexs.shape, valid_y.shape))
    valid_indexs = sequence.pad_sequences(valid_indexs, maxlen=max_features)
    # print(train_indexs.shape)

    print("Build model...")
    model = Sequential()
    # 此处包含下标0, 因此必须+1
    model.add(Embedding(len(words) + 1,
                        16,
                        input_length=max_features))
    model.add(Dropout(0.2))
    model.add(Conv1D(100,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    # 多分类构建 比如4分类
    # model.add(Dense(units=4, activation='softmax'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # 多分类 loss
    # model.compile(loss='categorical_crossentropy',
    #           optimizer='sgd',
    #           metrics=['accuracy'])
    # 优化器参数
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #           optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
    model.fit(train_indexs, y,
              batch_size=32,
              epochs=16,
              validation_data=(valid_indexs, valid_y))
    model.save_weights('hotel_cnn_model.h5')

    import pickle
    with open('word_indexs.dat', 'wb') as f:
        pickle.dump(word_indexs, f)
    with open('words.dat', 'wb') as f:
        pickle.dump(words, f)


    # 手动验证
    p = model.predict(valid_indexs)
    print(p)
    p = numpy.int64(p > 0.5)
    corrent = numpy.int64(p == valid_y.reshape(valid_y.size, 1)).sum()
    print(f"acc: {corrent / p.shape[0]}")
    # 0.85左右

