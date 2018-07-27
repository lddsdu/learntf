#-*- coding: utf-8 -*-
# @Time     :  下午2:16
# @Author   : ergouzi
# @FileName : dssm.py


import pandas as pd
from scipy import sparse
import collections
import random
import time
import numpy as np
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS
FLAGS.DEFINE_string("summaries_dir", "Summaries", "summaries directory")
