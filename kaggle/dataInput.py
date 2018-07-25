#-*- coding: utf-8 -*-
# @Time     :  下午1:01
# @Author   : ergouzi
# @FileName : dataInput.py

import tensorflow as tf
import os
import numpy as np


img_width = 224
img_height = 224


def get_file(file_dir):
    """
    获得图片的绝对地址以及图片对应的label返回
    :param file_dir:
    :return:
    """
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))

    labels = []
    for one_folder in images:
        letter = one_folder.split('/')[-1]
        if 'cat' in letter:
            labels.append(0)
        else:
            labels.append(1)

    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list


def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):
    """
    获得输入数据的batch
    :param image_list: 原始的image_list
    :param label_list:
    :param img_width:
    :param img_height:
    :param batch_size:
    :param capacity:
    :return:  返回image, label 的一个batch   tf.train.batch为什么就能返回批量的数据
    """
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
    image = tf.image.per_image_standardization(image) # 将图片标准化
    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)
    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch


# 将label转换为numpy的表示，使用one-hot来表示，用于计算softmaxloss
def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels
