#!/usr/bin/env python
# coding=utf-8
# 使用卷积神经网络实现手写数字识别训练
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

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

# 加载样本集
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
# 获取样本占位符
x = tf.placeholder(tf.float32, [None, 784], name="train_x")

# 定义样本标签值占位符
y_ = tf.placeholder(tf.float32, [None, 10])


# 定义第一层卷积 patch卷积大小为:5*5 卷积在5*5的区域中算出32个特征作为输入，因为是灰度图像所以通道数为1 rgb彩色图像应为3
W_conv1 = weight_variable([5, 5, 1, 32])
# 偏置单元个数和特征数一致
b_conv1 = bias_variable([32])

# 为了使用卷积层 将图像修改为4d张量 最后一个还是通道数 因为样本个数不固定 所以第一个参数是-1
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 然后对图像进行卷积 激活函数使用ReLU函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 对卷积结果执行max pooling
h_pool1 = max_pool_2x2(h_conv1)

# 定义第二层卷积 对5*5 产生64的特征输出 输入通道是上面的输出因为是灰度:1*32
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# 执行卷积和池化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 注意卷积不改变图像大小 但是池化会缩减图像 开始28*28 第一层池化:28*28/2*2 = 14*14 第二层池化:14*14/2*2 = 7*7 也就是目前图片大小达到7*7
# 最后输出也可以这样计算: 某一维度原来是N，经过一层卷积和池化:N = (N - F + 2*P)/(S + 1) F是卷积核对应维度长度，P是max pool长度 S是步长 对于多层就计算多次 小数直接取整不用四舍五入
# 构建密集连接层 特征总数:7*7*64 这里加入一个有1024个神经元的全连接层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# 调整第二层池化层输出张量 乘以权重矩阵 加上偏置单元 最后再使用ReLU计算
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合 一般在输出之前加入dropout 建议训练过程中使用dropout 但是在测试过程中关闭dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 最后添加一层输出层 相当于softmax回归一样 汇总为10维向量输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 训练和评估模型 求均值 下面方式可以提高性能
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_setp = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 比较测试结果和标签结果是否一致
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
# 开始执行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练3000次 每次随机取50张
    for i in range(3000):
        batch = mnist.train.next_batch(50)
        # 训练和测试中的keep_prob 表示dropout的比例
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_:batch[1], keep_prob: 1.0
                })
            print "step %d train_accuracy: %f" % (i, train_accuracy)
        train_setp.run(feed_dict={x: batch[0], y_:batch[1], keep_prob:0.5})

    print "test accuracy: %g" % (accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
        }))
    # 保存模型 训练3000批最终成功率能达到97%以上
    saver.save(sess, 'mnist_model/mnist.model')


