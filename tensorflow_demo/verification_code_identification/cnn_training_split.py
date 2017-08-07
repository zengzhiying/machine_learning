#!/usr/bin/env python
# coding=utf-8
# 构建卷积神经网络训练验证码图片 将验证码分割识别 减少训练的次数
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

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

# 彩色图片转成灰度
def convert_to_gray(image):
    return np.dot(image[..., :3], [0.2989, 0.587, 0.144])

# 随机获取一张验证码图片 并返回
def get_random_image(image_files):
    image_file = image_files[random.randint(0, len(image_files) - 1)]
    text = image_file[:-4].split('_')[1]
    image = Image.open('./train_images/%s' % image_file)
    image = np.array(image)
    return text, image

# 将普通图片保存为灰度图片
def convert_rgb_to_gray(image_file, output_image_file):
    img = Image.open(image_file).convert('L')
    img.save(output_image_file)

# 文本转为多分类的向量 那个字符出现就取哪个下标为1 向量一共包含4段 每一段代表一个验证码 顺序:[数字, 大写, 小写]
def text_to_vector(text):
    code_length = 10 + 26 + 26
    vector = np.zeros(4*code_length)
    for i, c in enumerate(text):
        index = ord(c)
        if index in range(48, 58):
            vector[i*code_length + index - 48] = 1
        elif index in range(65, 91):
            vector[i*code_length + index - 65 + 10] = 1
        else:
            vector[i*code_length + index - 97 + 36] = 1

    return vector


# 向量转回文本
def vector_to_text(vector):
    code_length = 10 + 26 + 26
    text = []

    # 由索引转字符
    def index_to_char(number):
        if number < 10:
            return str(number)
        if number < 36:
            char_ascii = number - 10 + 65
            return chr(char_ascii)
        char_ascii = number - 36 + 97
        return chr(char_ascii)

    for i, v in enumerate(vector):
        if v == 1:
            if i < code_length:
                text.append(index_to_char(i))
            elif i < 2*code_length:
                text.append(index_to_char(i - code_length))
            elif i < 3*code_length:
                text.append(index_to_char(i - 2*code_length))
            else:
                text.append(index_to_char(i - 3*code_length))

    return ''.join(text)

# 单文本字符转向量
def character_to_vector(character):
    code_length = 10 + 26 + 26
    vector = np.zeros(code_length)
    index = ord(character)
    if index in range(48, 58):
        vector[index - 48] = 1
    elif index in range(65, 91):
        vector[index - 65 + 10] = 1
    else:
        vector[index - 97 + 36] = 1
    return vector

# 单字符向量转回文本
def vector_to_character(vector):
    code_length = 10 + 26 + 26
    def index_to_char(number):
        if number < 10:
            return str(number)
        if number < 36:
            char_ascii = number - 10 + 65
            return chr(char_ascii)
        char_ascii = number - 36 + 97
        return chr(char_ascii)
    for i, v in enumerate(vector):
        if v == 1:
            return index_to_char(i)

# 随机生成一个训练batch
def get_next_batch(image_files, batch_size=128):
    batch_x = np.zeros([batch_size, 2700])
    batch_y = np.zeros([batch_size, 62])

    for i in range(batch_size):
        text, image = get_random_image(image_files)
        image = convert_to_gray(image)

        # 一维化并二值化
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = character_to_vector(text)

    return batch_x, batch_y

# 定义验证码cnn模型
def verify_code_cnn():
    # 样本
    X = tf.placeholder(tf.float32, [None, 2700])
    # 标签
    y_ = tf.placeholder(tf.float32, [None, 62])
    # 样本转换
    x = tf.reshape(X, shape=[-1, 60, 45, 1])

    # 第一层 卷积层
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层 卷积层
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 第三层 卷积层
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # 全连接层
    W_fc1 = weight_variable([8*6*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*6*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # dropout层
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    W_fc2 = weight_variable([1024, 62])
    b_fc2 = bias_variable([62])

    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return X, y_, y, keep_prob

# 训练验证码cnn
def train_cnn(image_files):
    X, y_, conv, keep_prob = verify_code_cnn()
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=conv))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # predict = tf.reshape(conv, [-1, 62, 4])

    # 按照行来取值 2表示按列
    max_o = tf.argmax(conv, 1)
    max_y = tf.argmax(y_, 1)

    correct_prediction = tf.equal(max_o, max_y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # 继续训练之前的模型
        saver.restore(sess, 'model/verify_code.model')
        step = 0
        while True:
            print "获取批量数据集."
            batch_x, batch_y = get_next_batch(image_files, 100)
            print "开始训练."
            train_step.run(feed_dict={X: batch_x, y_: batch_y, keep_prob: 0.75})
            if step % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={X: batch_x, y_:batch_y, keep_prob:1})
                print "step: %d train accuracy: %f" % (step, train_accuracy)
                if train_accuracy > 0.96:
                    saver.save(sess, 'model/verify_code.model')
                    break
            step += 1



if __name__ == '__main__':
    # 加载所有图片样本列表
    image_files = []
    for parent, dirnames, filenames in os.walk('./train_images'):
        for image_file in filenames:
            image_files.append(image_file)
            # print image_file

    print "加载完毕. 图片张数: %d" % len(image_files)
    text, image = get_random_image(image_files)
    print text
    print image.shape
    print convert_to_gray(image).shape

    vec = character_to_vector('M')
    print vec
    print vector_to_character(vec)

    # convert_rgb_to_gray('images/bfb39a7a78ca11e79cc0000c29463d8b_UJg0.jpg', './test.jpg')

    batch_x, batch_y = get_next_batch(image_files, 20)
    print batch_x.shape
    print batch_y.shape

    train_cnn(image_files)
