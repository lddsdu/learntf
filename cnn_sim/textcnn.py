#-*- coding: utf-8 -*-
# @Time     :  下午8:49
# @Author   : ergouzi
# @FileName : cnn_model.py


import tensorflow as tf
import numpy as np
from data_util import build_glove_dict

class TextCnn(object):
    """
    定义网络结构 网络的结构是什么样的？？？
    """
    def __init__(self, sequence_length, num_classes, filter_sizes, num_filters, l2_reg_lambda = 0.0):
        self.input_s1 = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length], name="input_s1")
        self.input_s2 = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length], name="input_s2")
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="input_y")
        self.drop_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")   # 这里直接申明为一个常数了么？？？
        self.filter_sizes = filter_sizes
        self.sequence_length = sequence_length
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = tf.constant(0.0)

        # 构建网络
        self.init_weight()
        self.inference()
        self.add_dropout()
        self.add_output()
        self.add_loss()
        self.add_acuracy()
        pass


    # 卷积层
    def conv(self, name, input_data, filter_shape, output_channel, activation = True):
        """
        卷积层  这里对于文本的卷积可能是不适用的了
        :param name:
        :param input_data:
        :param filter_shape: 卷积核的形状
        :param output_channel:
        :return:
        """
        in_channel = input_data.get_shape()[-1]
        filter_shape.append(in_channel)
        filter_shape.append(output_channel)
        weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name=name+ "-weight")
        biases = tf.Variable(tf.constant(0.1, shape=[output_channel], name=name + "-bias"))
        conv = tf.nn.conv2d(
            input_data,
            weight,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name=name
        )
        # nonlinerarity
        if activation:
            h = tf.nn.relu(tf.nn.bias_add(conv, biases), name="relu")
            return h
        else:
            return conv

    # maxpool 池化层
    def maxpool(self, name, input_data):
        """
        最大池化  没有参数，相对比较简单
        :param name:
        :param input_data:
        :return:
        """
        pool = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name=name)
        return pool

    def init_weight(self):
        # 返回值为 word2id 和　wordembedding
        _, self.word_embedding = build_glove_dict()
        self.W = tf.get_variable(name='word_embedding', shape=self.word_embedding.shape, dtype=tf.float32,
                                 initializer=tf.constant_initializer(self.word_embedding), trainable=True)
        self.embedding_size = self.word_embedding.shape[1]
        self.s1 = tf.nn.embedding_lookup(self.W, self.input_s1)
        self.s2 = tf.nn.embedding_lookup(self.W, self.input_s2)

        """
        >>> a = np.array([[1,1,1],[2,2,2]])
        >>> b = np.array([[3,3,3],[4,4,4]])
        >>> c = tf.concat([a, b], axis=0)
        >>> d = tf.concat([a, b], axis=1)
        >>> sess.run([c, d])
        [array([[1, 1, 1],
               [2, 2, 2],
               [3, 3, 3],
               [4, 4, 4]]), array([[1, 1, 1, 3, 3, 3],
               [2, 2, 2, 4, 4, 4]])]
        >>> 
        """
        self.x = tf.concat([self.s1, self.s2], axis = 1)
        # 作为网络的输入
        self.x = tf.expand_dims(self.x, -1)   # 最后得到的ｃｏｎｃａｔ数据增加一个维度什么作用

    def inference(self):
        """
        构建一个卷积　＋　最大池化　网络　　这里的卷积层完全写错了，不同于图像的卷积，
        文本的卷积没有很深层次的卷积，在ｄｅｅｐｒａｎｋ论文中也只是一层的卷积,目前对于
        这里的维度不是很理解
        :return:
        """
        self.pool_output = []

        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                bias   = tf.Variable(tf.constant(0.1, shape=[self.num_filters]),name="b")

                # def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,
                #            data_format=None, name=None):
                # conv2d的参数

                conv = tf.nn.conv2d(
                    self.x,
                    weight,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )

                h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")

                # def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
                pool = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool"
                )

                self.pool_output.append(pool)

        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(self.pool_output, 3)            # 对于第三个维度的ｃａｎｃａｔ
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

    def add_dropout(self):
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.pool_last_flat, self.drop_keep_prob)

    def add_output(self):
        """tf.nn.wx_plus_b相当于全连接层么"""
        with tf.name_scope("output"):
            W=tf.get_variable(
                "w",
                shape=[self.num_filters_total, self.num_classes],
                initializer=tf.global_variables_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes], name='b'))
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, 1, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    def add_loss(self):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

    def add_acuracy(self):
        with tf.name_scope("pearson"):
            mid1 = tf.reduce_mean(self.scores * self.input_y) - \
            tf.reduce_mean(self.scores) * tf.reduce_mean(self.input_y)

            mid2 = tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores))) * \
                   tf.sqrt(tf.reduce_mean(tf.square(self.input_y)) - tf.square(tf.reduce_mean(self.input_y)))

            self.pearson = mid1 / mid2





















