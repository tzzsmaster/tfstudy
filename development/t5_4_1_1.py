# -*- coding: utf-8 -*-
"""
@author: tz_zs

持久化实现 模型的保存和加载111
"""
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result1 = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # 保存模型到/path/to/model/model.ckpt文件
    saver.save(sess, "/path/to/model/model.ckpt")

'''
保存模型到了/path/to/model/model.ckpt文件中。
此时会生成三个文件：
model.ckpt.meta
model.ckpt
checkpoint
'''

# 加载模型
saver2 = tf.train.import_meta_graph("/path/to/model/model.ckpt.meta")

with tf.Session() as sess:
    saver2.restore(sess, "/path/to/model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))  # [ 3.]
