#-*- coding: utf-8 -*-
# @Time     :  下午2:48
# @Author   : ergouzi
# @FileName : minst_test.py
import tensorflow as tf

save_file = "/home/jack/Desktop/mnist_weights.ckpt1"
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, save_file)

if __name__ == '__main__':
    sess.run(y)