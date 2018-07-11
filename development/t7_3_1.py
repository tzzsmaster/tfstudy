# -*- coding: utf-8 -*-
"""
@author: tz_zs

多线程队列
"""
import tensorflow as tf

# 队列
queue = tf.FIFOQueue(100, "float")
# 入队
enqueue_op = queue.enqueue([tf.random_normal([1])])
# 多线程
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
# 将QueueRunner加入集合
tf.train.add_queue_runner(qr)
# 出队
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 协同
    coord = tf.train.Coordinator()
    # 启动集合中的QueueRunner，返回线程列表
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(7): print(sess.run(out_tensor)[0])

    coord.request_stop()
    coord.join(threads)
