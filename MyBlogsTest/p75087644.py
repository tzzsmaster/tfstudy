# -*- coding: utf-8 -*-
"""
@author: tz_zs

变量 初始值为常数的情况
"""
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0), name="v2")
v3 = tf.Variable(2.0, name="v3")
result1 = v1 + v2
result2 = v1 + v3
print(v1)  # <tf.Variable 'v1:0' shape=(1,) dtype=float32_ref>
print(v2)  # <tf.Variable 'v2:0' shape=() dtype=float32_ref>
print(v3)  # <tf.Variable 'v3:0' shape=() dtype=float32_ref>
print(result1)  # Tensor("add:0", shape=(1,), dtype=float32)
print(result2)  # Tensor("add_1:0", shape=(1,), dtype=float32)

with tf.Session() as sess:
    tf.global_variables_initializer().run()  # sess.run(tf.global_variables_initializer())
    print(sess.run(v1))  # [ 1.]
    print(sess.run(v2))  # 2.0
    print(sess.run(v3))  # 2.0
    print(sess.run(result1))  # [ 3.]
    print(sess.run(result2))  # [ 3.]
