#-*- coding: utf-8 -*-
# @Time     :  下午8:41
# @Author   : ergouzi
# @FileName : model_train.py

# 这里在import时的习惯，将第三方库先引入，然后在引入自己编写的文件

import datetime
import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

from textcnn import *
from data_util import *


# 参数
FLAGS = tf.app.flags

# DEFINE_XXX 带了三个参数，分别是变量名称，默认值，用法描述  这里原来代表的是默认值，可以通过运行时的命令更改参数
# 主要用于获取参数，设置默认的参数

# 数据参数
tf.app.flags.DEFINE_float("train_sample_percentage", 0.9, "Percentage of the train data to use for validation")
tf.app.flags.DEFINE_string("data_file", "SICK_data/SICK.txt", "Data source")
tf.app.flags.DEFINE_string("data_path","/home/jack/Downloads/hdd_download/TF_Sentence_Similarity_CNN-master","data path")

# 模型超参数
tf.app.flags.DEFINE_string("filter_size", "3, 4, 5", "comma-seprated filter size (default: '3, 4, 5')")
tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.app.flags.DEFINE_integer("seq_length", 36, "sequence length (default: 36)")
tf.app.flags.DEFINE_integer("num_classes", 1, "number_classes (default: 1)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probablity (default: 0.5)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularization lambda (default: 1)")

# 训练参数
tf.app.flags.DEFINE_integer("batch_size", 64, "batch_size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many step (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100, " (Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoint to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 获取训练数据  数据的顺序已经打乱了
s1, s2, score = read_data_sets(FLAGS.data_file)
score = np.asarray([[s] for s in score])
sample_num = len(score)
train_end = int(sample_num * FLAGS.train_sample_percentage)

# 将数据划分为训练数据与测试数据
s1_train, s1_test = s1[:train_end], s1[train_end:]
s2_train, s2_test = s2[:train_end], s2[train_end:]

score_train, score_test = score[:train_end], score[train_end:]
print "train/test split {:d}/{:d}".format(len(score_train), len(score_test))


sess = tf.Session()

# __init__(self, sequence_length, num_classes, filter_sizes, num_filters,
#  l2_reg_lambda = 0.0)
textcnn = TextCnn(
    sequence_length=FLAGS.seq_length,
    num_classes=FLAGS.num_classes,
    filter_size=map(int, FLAGS.filter_sizes.split(",")),
    num_filters=FLAGS.num_filters,
    l2_reg_lambda=FLAGS.l2_reg_lambda
)

"""
几种常见的优化器:各种optimizer的权重更新的策略不同
Optimizer 
GradientDescentOptimizer 
AdagradOptimizer 
AdagradDAOptimizer 
MomentumOptimizer 
AdamOptimizer 
FtrlOptimizer 
RMSPropOptimizer
"""

# define training procedure
global_step = tf.Variable(0, name="global_step", trainable=False)  # global_step 记录全局训练步骤的单值, 其值一直会自增么???
optimizer = tf.train.AdamOptimizer(1e-3)
# 这里为什么自己来算梯度
grads_and_vars = optimizer.compute_gradients(textcnn.loss)

"""global_step: Optional `Variable` to increment by one after the
        variables have been updated."""
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# 跟踪 gradient values and sparsity
grad_summaries = []

for g, v in grads_and_vars:
    if g is  not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparisty".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.summary.merge(grad_summaries)

# summary for loss and pearson
loss_summary = tf.summary.scalar("loss", textcnn.loss)
acc_summary = tf.summary.scalar("pearson", textcnn.pearson)

# train summary
train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
out_dir = "/home/jack/Desktop"    #
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# dev summary
dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

# checkpoint directory, tensorflow assumes that directory already exist so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoint"))
checkpoint_prefix = os.path.join(checkpoint_dir, "textcnn-model")

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

"""
  def __init__(self,
               var_list=None,
               reshape=False,
               sharded=False,
               max_to_keep=5,
               keep_checkpoint_every_n_hours=10000.0,
               name=None,
               restore_sequentially=False,
               saver_def=None,
               builder=None,
               defer_build=False,
               allow_empty=False,
               write_version=saver_pb2.SaverDef.V2,
               pad_step_number=False,
               save_relative_paths=False,
               filename=None):
"""
# 貌似以前的saver没有使用任何的参数
saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)


# 初始化参数
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def train_step(s1, s2, score):
    """
    A single training step  # 返回一个batch???
    :param s1:
    :param s2:
    :param score:
    :return:
    """
    feed_dict = {
        textcnn.s1 : s1,
        textcnn.s2 : s2,
        textcnn.scores : score,
        textcnn.drop_keep_prob : FLAGS.dropout_keep_prob
    }

    # 哪些数据是要进行计算的？？？ loss accuracy summary global_step train_op
    _, step,summaries, loss, pearson = sess.run([train_op,
              global_step,
              train_summary_op,
              textcnn.loss,
              textcnn.pearson
              ], feed_dict=feed_dict)
    train_summary_writer.add_summary(summaries, step)


def dev_step(s1, s2, score, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        textcnn.s1: s1,
        textcnn.s2: s2,
        textcnn.input_y: score,
        textcnn.drop_keep_prob: 1.0     # 这里设置的是不丢弃任何参数
    }

    step, summaries, loss, pearson = sess.run([
        global_step, dev_summary_op, textcnn.loss, textcnn.pearson
    ], feed_dict=feed_dict)

    if writer:
        writer.add_summary(summaries, step)

STS_train = dataset(s1=s1_train, s2=s2_train, label=score_train)

for i in range(40000):
    batch_train = STS_train.next_batch(FLAGS.batch_size)
    train_step(batch_train[0], batch_train[1], batch_train[2])
    # 用于获取当前的step么， 前面不是有了i么
    current_step = tf.train.global_step(sess, global_step)

    # 吐槽一下这里的变量的命名的方式，xxx_every，这个也太乡土了吧。。。 interval
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        dev_step(s1_test, s2_test, score_test, writer=dev_summary_writer)
        print ""

    if current_step % FLAGS.checkpoint_every == 0:
        print "\nSave checkpoint"
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("")






















