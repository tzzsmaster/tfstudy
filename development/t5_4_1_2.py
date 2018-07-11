# -*- coding: utf-8 -*-
"""
@author: tz_zs

滑动平均值的存储和加载（持久化）113114
"""
import tensorflow as tf

v1 = tf.Variable(10, dtype=tf.float32, name="v1")

for variables in tf.global_variables():  # all_variables弃用了
    print(variables)  # <tf.Variable 'v1:0' shape=() dtype=float32_ref>

ema = tf.train.ExponentialMovingAverage(0.99)
print(ema)  # <tensorflow.python.training.moving_averages.ExponentialMovingAverage object at 0x00000218AE5720F0>

maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables():
    print(variables)
    # <tf.Variable 'v1:0' shape=() dtype=float32_ref>
    # <tf.Variable 'v1/ExponentialMovingAverage:0' shape=() dtype=float32_ref>

saver = tf.train.Saver()
print(saver)  # <tensorflow.python.training.saver.Saver object at 0x0000026B7E591940>
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(tf.assign(v1, 1))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))  # [1.0, 9.9099998]

    print(saver.save(sess, "/path/to/model.ckpt"))  # 持久化存储____会返回路径 /path/to/model.ckpt
#################################################################################################
print("#####" * 10)
print("加载")
#################################################################################################
var2 = tf.Variable(0, dtype=tf.float32, name="v2")
print(var2)  # <tf.Variable 'v2:0' shape=() dtype=float32_ref>
saver2 = tf.train.Saver({"v1/ExponentialMovingAverage": var2})
with tf.Session() as sess2:
    saver2.restore(sess2, "/path/to/model.ckpt")
    print(sess2.run(var2))  # 9.91 所以，成功加载了v1的滑动平均值
'''
var3 = tf.Variable(0, dtype=tf.float32, name="v1")
print(var3)  # <tf.Variable 'v1:0' shape=() dtype=float32_ref>
ema = tf.train.ExponentialMovingAverage(0.99)

print(ema.variables_to_restore())  # {'v1/ExponentialMovingAverage': <tf.Variable 'v1:0' shape=() dtype=float32_ref>}
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "/path/to/model.ckpt")
    print(sess.run(var3))  # 9.91
'''