# -*- coding: utf-8 -*-
"""
@author: tz_zs
"""
import numpy as np
import tensorflow as tf

# a = tf.random_normal([1])
# with tf.Session() as sess:
#     b = sess.run(a)
#     print(a)
#     print(b)

v1 = tf.Variable(1, dtype=tf.float32)
v2 = tf.Variable([2], dtype=tf.float32)
v3 = tf.Variable(tf.constant(3.0), dtype=tf.float32)
v4 = tf.Variable(tf.constant([2.0]), dtype=tf.float32)
print(v4==v2)
print(v1)
print(v2)
print(v3)
print(v4)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(v1.eval())
    print(v2.eval())
    print(v3.eval())
    print(v4.eval())
