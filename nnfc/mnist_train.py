# -*- coding: utf-8 -*-
"""
@author: tz_zs

MNIST 升级----mnist_train.py
"""
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载函数
from nnfc import mnist_inference

# 配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存路径和文件名
MODEL_SAVE_PATH = "/path/to/model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    # 定义输入输出的placeholder
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    # 定义正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 使用前向传播
    y = mnist_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 优化算法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 持久化
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 运行
            _, loss_valuue, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            # 每1000轮保存一次模型
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_valuue))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

'''
After 1 training step(s), loss on training batch is 3.22362.
After 1001 training step(s), loss on training batch is 0.202338.
After 2001 training step(s), loss on training batch is 0.141154.
After 3001 training step(s), loss on training batch is 0.13816.
After 4001 training step(s), loss on training batch is 0.123687.
After 5001 training step(s), loss on training batch is 0.116358.
After 6001 training step(s), loss on training batch is 0.0994073.
After 7001 training step(s), loss on training batch is 0.0853637.
After 8001 training step(s), loss on training batch is 0.0775001.
After 9001 training step(s), loss on training batch is 0.072494.
After 10001 training step(s), loss on training batch is 0.0755896.
After 11001 training step(s), loss on training batch is 0.0617309.
After 12001 training step(s), loss on training batch is 0.0621173.
After 13001 training step(s), loss on training batch is 0.0540873.
After 14001 training step(s), loss on training batch is 0.0491002.
After 15001 training step(s), loss on training batch is 0.0505174.
After 16001 training step(s), loss on training batch is 0.0451144.
After 17001 training step(s), loss on training batch is 0.0472387.
After 18001 training step(s), loss on training batch is 0.041461.
After 19001 training step(s), loss on training batch is 0.0393669.
After 20001 training step(s), loss on training batch is 0.0477065.
After 21001 training step(s), loss on training batch is 0.0442965.
After 22001 training step(s), loss on training batch is 0.0363835.
After 23001 training step(s), loss on training batch is 0.0386328.
After 24001 training step(s), loss on training batch is 0.0365634.
After 25001 training step(s), loss on training batch is 0.0398796.
After 26001 training step(s), loss on training batch is 0.0374554.
After 27001 training step(s), loss on training batch is 0.034578.
After 28001 training step(s), loss on training batch is 0.0341904.
After 29001 training step(s), loss on training batch is 0.0366765.
'''