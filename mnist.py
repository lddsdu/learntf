# -*- coding: utf-8 -*-
import struct
import numpy as np
import os
import tensorflow as tf

def read_train_image(filename):
    index = 0
    binfile = open(filename, "rb")
    buf = binfile.read()
    magic, train_image_num, num_rows, num_colums = struct.unpack_from(">IIII", buf, index)
    train_image_list = np.zeros((train_image_num, 28 * 28))
    index += struct.calcsize('>IIII')
    for i in range(train_image_num):
        im = struct.unpack_from(">784B", buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im)
        im = im / 255
        im = im.reshape(1, 28 * 28)
        train_image_list[i, :] = im
    return train_image_list


def read_train_label(filename):
    index = 0
    binfile = open(filename, "rb")
    buf = binfile.read()
    magic, train_label_num = struct.unpack_from(">II", buf, index)
    train_label_list = np.zeros((train_label_num, 10))
    index += struct.calcsize(">II")
    for i in range(train_label_num):
        labelTemp = np.zeros(10)
        label = struct.unpack_from(">1B", buf, index)
        index += struct.calcsize(">1B")
        label = np.array(label)
        labelTemp[label[0]] = 1
        train_label_list[i, :] = labelTemp
    return train_label_list


#随机的添加batch进行训练
def next_batch_image(batch_count, image_list, label_list):
    rnd = np.random.randint(1, 60000 - batch_count)
    return image_list[rnd:rnd+batch_count], label_list[rnd:rnd+batch_count]


def read_test_image(filename):
    return read_train_image(filename)


def read_test_label(filename):
    return read_train_label(filename)

mnist = "/home/jack/train_data/mnist"
train_image_filename = os.path.join(mnist, "train-images-idx3-ubyte")
test_image_filename = os.path.join(mnist, "t10k-images-idx3-ubyte")
train_label_filename = os.path.join(mnist, "train-labels-idx1-ubyte")
test_label_filename = os.path.join(mnist, "t10k-labels-idx1-ubyte")


#获取到数据集
train_image_list = read_train_image(train_image_filename)
train_label_list = read_train_label(train_label_filename)
test_image_list = read_test_image(test_image_filename)
test_label_list = read_test_label(test_label_filename)


# 训练样本image placeholder 是 n * 784
x = tf.placeholder("float", [None, 784])
# 权重
W = tf.Variable(tf.zeros([784, 10]))
# bias
b = tf.Variable(tf.zeros([10]))
# 训练结果
y = tf.nn.softmax(tf.matmul(x,W) + b)
# 交叉熵
y_ = tf.placeholder("float", [None, 10])
# 设置的损失函数？？？
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 梯度下降法,最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
# 变量的初始化
sess.run(init)


#进行训练
for i in range(3000):
    batch_xs, batch_ys = next_batch_image(128,train_image_list, train_label_list)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
    if i % 100 == 0:
        # 用来测试识别准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print (sess.run(accuracy, feed_dict={x: test_image_list, y_: test_label_list}))



