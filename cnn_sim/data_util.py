#-*- coding: utf-8 -*-
# @Time     :  下午7:10
# @Author   : ergouzi
# @FileName : data_util.py

from variable_consant import *
import os
import word2vec
import numpy as np
import pandas as pd
from multiprocessing import Pool
import re

# 定义全局变量
sr_word2id = None
word_embedding = None
sample_num = None


def clean_str(string):
    # 对句子相似度任务进行字符清洗
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_glove_dict(filename = "glove.6B.50d.txt"):
    """
    从文件读取词向量，使用到了word2vec 和　pd这两个库
    对于在sr_word2id中不存在的文件，则使用了<unk>来表示
    :param filename:
    :return:
    """
    wv = word2vec.load(os.path.join(glove_path, filename))
    vocab = wv.vocab
    sr_word2id = pd.Series(range(1, len(vocab) + 1), index=vocab)
    sr_word2id["<unk>"] = 0
    word_embedding = wv.vectors
    word_mean = np.mean(word_embedding, axis=0)
    word_embedding = np.vstack([word_mean, word_embedding])
    return sr_word2id, word_embedding


def get_id(word):
    """将单个单词转化为ｉｄ来表示"""
    if word2vec in sr_word2id:
        return sr_word2id[word]
    else:
        return sr_word2id["<unk>"]


def seq2id(sentence):
    """将seq转化为id表示"""
    sentence = clean_str(sentence)
    sentence_split = sentence.split(' ')
    """这里ｍａｐ使用到了函数式编程的思想，这里的get_id当做了参数直接传递进来"""
    seq_id = map(get_id, sentence_split)
    return seq_id


def padding_sentence(s1, s2):
    """对句子进行填充 寻找最长的句子，将其他的句子填充０"""
    s1_max_len = max([len(s) for s in s1])
    s2_max_len = max([len(s) for s in s2])
    max_length = max(s1_max_len, s2_max_len)
    global sample_num
    sample_num = s1.shape[0]
    s1_encodding = np.zeros(shape=[sample_num, max_length])
    s2_encodding = np.zeros(shape=[sample_num, max_length])

    for i, s in enumerate(s1):
        s1_encodding[i][: len(s)] = s

    for i, s in enumerate(s2):
        s2_encodding[i][: len(s)] = s

    # 总共包含了 9840　个样例
    print "填充所有的句子完毕"
    return s1_encodding, s2_encodding


def read_data_sets(train_dir):
    """
    s1 代表了数据集的句子 1
    s2 代表了数据集的句子 2
    score 代表了 s1 s2之间的相关度
    sample_num 代表了总共的样例数量
    :param train_dir:
    :return:
    """
    sickfile = os.path.join(sick_data_dir, sick_file_name)
    df_sick = pd.read_csv(sickfile, sep="\t", usecols=[1, 2, 4], names=
                          ['s1', 's2', 'score'], dtype={'s1':object,
                          's2':object, 's3': object})
    df_sick = df_sick.drop([0])
    s1 = df_sick.s1.values
    s2 = df_sick.s2.values
    score = np.array(map(float, df_sick.score.values), dtype=np.float32)
    sample_num = len(score)

    global sr_word2id, word_embedding
    sr_word2id, word_embedding = build_glove_dict()

    # 1,将ｗｏｒｄ转换为id  使用pool.map可以直接使用到多个线程
    pool = Pool(6)
    """
    >>> a=np.asarray([[1,　2, 4],　[3,　4]])
    >>> a
    array([list([1, 2, 4]), list([3, 4])], dtype=object)
    这里asarray会尽量深入的将数据转化为ndarray

    """
    s1 = np.asarray(pool.map(seq2id, s1))
    s2 = np.asarray(pool.map(seq2id, s2))
    # close the pool and wait the worker to exit
    pool.close()
    pool.join()

    # 2,填充句子
    s1, s2 = padding_sentence(s1, s2)
    new_index = np.random.permutation(sample_num)
    """这里 s1[new_index]是ｎｕｍｐｙ的ａｒｒａｙ所特有的方法"""
    s1 = s1[new_index]
    s2 = s2[new_index]
    score = score[new_index]

    """至此位置，所有的数据预处理结束"""
    return s1, s2, score


class dataset(object):
    """
    将所有的训练数据交给一个类来管理，这样的话，可以直接调用这个类的方法获得训练所需要的batch
    """
    def __init__(self, s1, s2, label):
        self.index_in_epoch  = 0
        self.s1 = s1
        self.s2 = s2
        self.label = label
        self.example_num = len(label)
        self.epoche_complete = 0

    def next_batch(self, batch_size):
        """策略： 为了获得batch简单考虑，在一个batch不够分配时，打乱所有数据顺序，从头开始，并记为一个epoch完成"""
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.example_num:
            self.epoche_complete += 1
            perm = np.random.permutation(self.example_num)
            self.s1 = self.s1[perm]
            self.s2 = self.s2[perm]
            self.label = self.label[perm]
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.example_num
        end = self.index_in_epoch
        return np.array(self.s1[start:end]), np.array(self.s2[start:end]), np.array(self.label[start:end])










