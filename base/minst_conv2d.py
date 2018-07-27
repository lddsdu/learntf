#-*- coding: utf-8 -*-
# @Time     :  下午5:09
# @Author   : ergouzi
# @FileName : minst_conv2d.py

import struct
import numpy as np
import os
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_dir = "/home/jack/train_data/mnist"


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
        im = im / 255.0
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


# 随机的添加batch进行训练
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


x = tf.placeholder(tf.float32, shape=([None, 784]), name="input_x")
y_ = tf.placeholder(tf.float32, shape=([None, 10]), name="label_y")     # 使用到softmaxloss 在写损失函数公式的时候需要用到

# 获取到数据集
train_image_list = read_train_image(train_image_filename)
train_label_list = read_train_label(train_label_filename)
test_image_list = read_test_image(test_image_filename)
test_label_list = read_test_label(test_label_filename)


# 定义函数，用于初始化所有的权重项w
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构建网络
x_image = tf.reshape(x, [-1, 28, 28, 1])                      #转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)     #第一个卷积层
h_pool1 = max_pool(h_conv1)                                  #第一个池化层

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      #第二个卷积层
h_pool2 = max_pool(h_conv2)                                   #第二个池化层

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])              #reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    #第一个全连接层

keep_prob = tf.placeholder(dtype=tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout层

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   #softmax层

# 设置的损失函数？？？
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_predict), reduction_indices=[1])
# 梯度下降法,最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
# 变量的初始化
sess.run(init)


#进行训练
for i in range(30000):
    batch_xs, batch_ys = next_batch_image(20, train_image_list, train_label_list)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
    if i % 1000 == 0:
        # 用来测试识别准确率
        correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_sum = 0
        for i in range(500):
            index = i * 20
            accuracy_sum += sess.run(accuracy, feed_dict={x: test_image_list[index: index+20], y_: test_label_list[index:index + 20], keep_prob: 1.0})
        print (accuracy_sum / 500)

saver = tf.train.Saver()
#注意这里save_path包含了最终生成的三个文件的前缀部分
saver.save(sess, data_dir+"/mnist_conv2d")
sess.close()

