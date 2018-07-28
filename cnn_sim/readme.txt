文本相似度
reference: https://blog.csdn.net/irving_zhang/article/details/69440789

tf.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularization lambda (default: 0.0)")
如果这里的L2_reg_lambda设置的较大，那么最后的loss将维持一个较大的值，这里l2_reg_lambda的主要作用为
防止过拟合, 但是为了最后的loss不能过大，所以设置的应该稍微小一点
