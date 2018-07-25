#-*- coding: utf-8 -*-
# @Time     :  下午12:25
# @Author   : ergouzi
# @FileName : VGG_16.py

import numpy as np
import tensorflow as tf


class vgg16():
    """
    模型结构定义： 使用了卷积，池化，全连接 还定义了权重是否进行训练
    其中fc, conv, maxpool用来定义网络的层，其中包含了权重等数据
    """
    def __init__(self, images):
        self.parameters = []
        self.images = images
        self.convlayers()
        self.fc_layers()
        self.probs = self.fc8

    def saver(self):
        return tf.train.Saver()

    def maxpool(self, name, input_data, trainable):
        """
        maxpool中不存在权重等数据，不需要进行训练
        :param name:
        :param input_data:
        :param trainable:
        :return:
        """
        out = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name=name)
        # 上述参数　[1, 2, 2, 1]表示了ksize　分别表示了在batch_size, height, width, channel中池化
        # 第二个 [1, 2, 2, 1] 表示了stride　分别表示了各个维度上的步长
        return out

    # 对于卷积层，需要知道输入的input_data，
    def conv(self, name, input_data, out_channel, trainable):
        """
        卷积层
        :param name:
        :param input_data:
        :param out_channel: 输出的维度
        :param trainable:
        :return:
        """
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32, trainable= False)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32, trainable=False)
            # f.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)  strides [batch, width, height, channel]
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            # 在卷积层后面直接接relu
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]
        return out

    def fc(self, name, input_data, out_channel, trainable=True):
        """
        全连接层
        :param name:
        :param input_data:
        :param out_channel: 需要知道目标维度是多少
        :param trainable:
        :return:
        """
        shape = input_data.get_shape().as_list()
        # 这个写法貌似是不正确的吧，最前面的一个维度留给了batch_size，后面的维度的变量全部用来进行全连接操作
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:
            size = shape[1]

        input_data_flat = tf.reshape(input_data, [-1, size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights", shape=[size, out_channel], dtype=tf.float32, trainable= trainable)
            biases = tf.get_variable(name="biases", shape=[out_channel], dtype=tf.float32, trainable= trainable)
            res = tf.matmul(input_data_flat, weights)
            out = tf.nn.relu(tf.nn.bias_add(res, biases))
        self.parameters += [weights, biases]
        return out

    def convlayers(self):
        # zero-mean input
        # conv1
        self.conv1_1 = self.conv("conv1re_1", self.images, 64, trainable=False)
        self.conv1_2 = self.conv("conv1_2", self.conv1_1, 64, trainable=False)
        self.pool1 = self.maxpool("poolre1", self.conv1_2, trainable=False)

        # conv2
        self.conv2_1 = self.conv("conv2_1", self.pool1, 128, trainable=False)
        self.conv2_2 = self.conv("convwe2_2", self.conv2_1, 128, trainable=False)
        self.pool2 = self.maxpool("pool2", self.conv2_2, trainable=False)

        # conv3
        self.conv3_1 = self.conv("conv3_1", self.pool2, 256, trainable=False)
        self.conv3_2 = self.conv("convrwe3_2", self.conv3_1, 256, trainable=False)
        self.conv3_3 = self.conv("convrew3_3", self.conv3_2, 256, trainable=False)
        self.pool3 = self.maxpool("poolre3", self.conv3_3, trainable=False)

        # conv4
        self.conv4_1 = self.conv("conv4_1", self.pool3, 512, trainable=False)
        self.conv4_2 = self.conv("convrwe4_2", self.conv4_1, 512, trainable=False)
        self.conv4_3 = self.conv("conv4rwe_3", self.conv4_2, 512, trainable=False)
        self.pool4 = self.maxpool("pool4", self.conv4_3, trainable=False)

        # conv5
        self.conv5_1 = self.conv("conv5_1", self.pool4, 512, trainable=False)
        self.conv5_2 = self.conv("convrwe5_2", self.conv5_1, 512, trainable=False)
        self.conv5_3 = self.conv("conv5_3", self.conv5_2, 512, trainable=False)
        self.pool5 = self.maxpool("poorwel5", self.conv5_3, trainable=False)

    def fc_layers(self):
        """
        两次全连接到4096 后再全连接到1000,　最后经过softmax
        :return:
        """
        self.fc6 = self.fc("fc6", self.pool5, 4096, trainable=False)
        self.fc7 = self.fc("fc7", self.fc6, 4096, trainable=False)
        # 前面的层是使用训练好了的权重，最后的fc则是最新的需要训练的权重
        self.fc8 = self.fc("fc8", self.fc7, 2)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i not in [30, 31]:
                # 这里使用的assign代表了给权重进行赋值，就是前面的get_variable
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----------all done---------------")