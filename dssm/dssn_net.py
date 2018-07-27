#-*- coding: utf-8 -*-
# @Time     :  上午9:11
# @Author   : ergouzi
# @FileName : dssn_net.py

import tensorflow as tf


class DSSM_NET():
    """
    定义ｄｓｓm网络的结构
    """
    fc_name_prefix = "fc%d"
    index = 0

    def __init__(self, input_data):
        self.input_data = input_data
        self.fc_layers()
        self.encoding_128 = self.fc3
        pass

    def add_layer(self, input_data, output_channel, name=fc_name_prefix%index):
        """
        全连接层　使用了三层全连接
        :param input_data:
        :param output_channel:
        :param name:
        :return:
        """
        global index
        index += 1
        shape = input_data.get_shape().as_list()
        # shape最前面的维度代表了batch_size
        size = 1
        for i in range(1, len(shape)):
            size *= shape[i]
        input_data_flat = input_data.reshape([-1, size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weight", shape=[size, output_channel], dtype=tf.float32, initializer=tf.random_uniform_initializer)
            biases = tf.get_variable(name="biases", shape=[output_channel], dtype=tf.float32, initializer=tf.random_uniform_initializer)
            res = tf.matmul(input_data_flat, weights)
            out = tf.nn.relu(tf.nn.bias_add(res, biases))
        return out

    def fc_layers(self):
        """
        构建三层全连接层
        :return:
        """
        self.fc1 = self.add_layer(self.input_data, 300)
        self.fc2 = self.add_layer(self.fc1, 300)
        self.fc3 = self.add_layer(self.fc2, 128)