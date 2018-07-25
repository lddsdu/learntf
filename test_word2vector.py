#-*- coding: utf-8 -*-
# @Time     :  上午10:36
# @Author   : ergouzi
# @FileName : test_word2vector.py
# 如何使用tensflow训练好的权重来进行预测   好尴尬，不知道怎么使用训练好的权重进行预测
import tensorflow as tf
import pickle


sess = tf.Session()
saver = tf.train.Saver()
weight_file = "/home/jack/train_data/word2vector/wv"
saver.restore(sess, weight_file)

filename = "/home/jack/train_data/word2vector/dict.txt"


def get_word_dict(filename):
    f = open(filename, "rb")
    pickle.load(f)
    f.close()


dictionary = get_word_dict(filename)


words = ["cat", "dog", "fish"]
input = tf.placeholder(tf.float32,[1])


for word in words:
    print("word = ", word)
    sess.run()


