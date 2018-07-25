#-*- coding: utf-8 -*-
# @Time     :  下午4:09
# @Author   : ergouzi
# @FileName : dog_or_cat.py
# 使用model_train训练获得的结果进行预测

import tensorflow as tf
import numpy as np
from dataInput import *
from scipy.misc import imread, imresize
import os
from VGG_16 import vgg16 as model


data_path = "/home/jack/train_data/kaggle/"
image_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = model(image_placeholder)
cat_or_dog = vgg.probs
sess = tf.Session()
saver = vgg.saver()
saver.restore(sess, "./model/dog_cat")

# image的路径
image_paths = []
for root, sub_folders, files in os.walk(os.path.join(data_path, "test1")):
    for name in files:
        image_paths.append(os.path.join(root, name))
image_paths = sorted(image_paths)


def read_image(image_path):
    data = imread(image_path)           # bytes
    data = imresize(data, (224, 224))     # tensor
    return data                         # ndarray


def softmax(nd):
    """
    对于指定维度的计算
    :param nd:
    :return:
    """
    res = np.zeros(nd.shape, dtype = nd.dtype)
    for i in range(res.shape[0]):
        temp = nd[i].copy()
        # 经过softmax获得的值
        temp = np.exp(temp) / np.sum(np.exp(temp), axis=0)
        res[i][...] = temp
    return res.astype(np.int32)


batch_size = 25
for i in range(len(image_paths) // batch_size):
    temp = []
    for j in range(batch_size):
        temp.append(read_image(image_paths[i * batch_size + j]))
    prediction = sess.run(cat_or_dog, feed_dict={image_placeholder:temp})
    print softmax(prediction)

sess.close()

