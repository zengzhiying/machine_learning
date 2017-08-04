#!/usr/bin/env python
# coding=utf-8
# tensorflow 基本用法
'''
步骤:
    1. 构建图 创建各种op(节点) 连接各op执行步骤 构成图 每个op可以产生0或者多个tensor(数据)
    2. 启动图 使用session来管理context 图必须在session上下文中汇总执行
    3. 执行图 调用session的run方法来执行具体的图计算 一般2,3步骤结合非常紧可以看成一步

    典型案例: 神经网络训练 在构建阶段构建一个图来表示以及训练神经网络，然后在执行阶段反复执行具体的训练op
'''
import tensorflow as tf

# 构建图 start
# 创建常量op 是一个1*2的矩阵 并加载到默认图 tf的方法返回值就是op的返回值
matrix1 = tf.constant([[3., 3.]])

# 构建另一个op 是一个2*1的矩阵
matrix2 = tf.constant([[2.], [2.]])

# 创建矩阵相乘op 把前面两个矩阵op作为输入 返回值代表这个op的输出
product = tf.matmul(matrix1, matrix2)
# 构建图 end

# 启动图
sess = tf.Session()

# 执行图 获得op输出 返回结果
result = sess.run(product)
print result

# 执行完毕 关闭会话
sess.close()

# 启动执行图部分可以使用with来完成
print "使用with."
with tf.Session() as sess:
    result = sess.run(product)
    print result


# 使用变量op
state = tf.Variable(0, name="counter")

# 创建常量op
one = tf.constant(1)

# 创建op 用于累加上面op
new_value = tf.add(state, one)

# 创建op 将new_value的值赋值给state
update = tf.assign(state, new_value)

# 使用变量必须先创建初始化op
# init_op = tf.initialize_all_variables() 此方法已经废弃 会警告不建议使用
init_op = tf.global_variables_initializer()

# 执行图
with tf.Session() as sess:
    # 运行init op
    sess.run(init_op)
    # 打印state的值 变量打印值必须用run来进行
    print sess.run(state)
    # 运行op 更新 state
    for i in range(3):
        print "new_value: %d" % sess.run(new_value)
        # 运行update 会自动运行调用 add op 并运行
        sess.run(update)
        print sess.run(state)

# 下面可以同时取回多个节点op的输出
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed) # 求乘积 mul方法已经废弃

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print result
    print result[0]
    print result[1]


# feed操作 用于临时替代tensor 在运行时提供feed数据作为参数 只使用一次 方法结束feed就消失了
# 标记方法是使用占位符
input1 = tf.placeholder(tf.float32) # tf.types已经废弃
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print sess.run([output], feed_dict={input1:[7.], input2:[2.]})
