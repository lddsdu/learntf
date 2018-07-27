#-*- coding: utf-8 -*-
# @Time     :  下午1:15
# @Author   : ergouzi
# @FileName : model_train.py

import tensorflow as tf
from VGG_16 import vgg16 as model
import os
import dataInput as reader
import time


data_path = "/home/jack/train_data/kaggle/"
X_train, Y_train = reader.get_file(os.path.join(data_path, "./train"))
# 使用了多个thread去加载数据
image_batch, label_batch = reader.get_batch(X_train, Y_train, 224, 224, 25, 256)
x_images_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3])          # 代表了输入的维度，但是没有将batch大小明确的写出来，使用None来表示
y_images_placeholder = tf.placeholder(tf.float32, [None, 2])                    # 这里就代表了二分类
vgg = model(x_images_placeholder)
# 将vgg model中的数据拿出来，用来sess.run()中计算
fc3_cat_and_dog = vgg.probs
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_cat_and_dog, labels=y_images_placeholder))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# 初始化各种参数
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 从文件中获得权重
vgg.load_weights("/home/jack/Downloads/hdd_download/vgg16_weights.npz", sess)
saver = vgg.saver()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
start_time = time.time()
for i in range(2000):
    image, label = sess.run([image_batch, label_batch])
    labels = reader.onehot(label)
    sess.run(optimizer, feed_dict={x_images_placeholder: image, y_images_placeholder: labels})
    loss_record = sess.run(loss, feed_dict={x_images_placeholder:image, y_images_placeholder:labels})
    print("loss ", i, " : ", loss_record)
    end_time = time.time()
    print("time used : ", (end_time - start_time))
    start_time = end_time


saver.save(sess, "./model/dog_cat")
print("Optimization done")




