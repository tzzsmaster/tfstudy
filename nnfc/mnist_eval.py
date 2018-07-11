# -*- coding: utf-8 -*-
"""
@author: tz_zs

测试程序
"""
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from nnfc import mnist_inference, mnist_train

# 每十秒加载一次最新的模型，并在测试数据上测试最新模型的准确率
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 测试(测试时不用计算正则化损失)
        y = mnist_inference.inference(x, None)

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 加载模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # print(variables_to_restore)
        # {'layer2/biases/ExponentialMovingAverage': < tf.Variable
        # 'layer2/biases:0'
        # shape = (10,)
        # dtype = float32_ref >, 'layer2/weights/ExponentialMovingAverage': < tf.Variable
        # 'layer2/weights:0'
        # shape = (500, 10)
        # dtype = float32_ref >, 'layer1/biases/ExponentialMovingAverage': < tf.Variable
        # 'layer1/biases:0'
        # shape = (500,)
        # dtype = float32_ref >, 'layer1/weights/ExponentialMovingAverage': < tf.Variable
        # 'layer1/weights:0'
        # shape = (784, 500)
        # dtype = float32_ref >}

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        while True:
            with tf.Session(config=config) as sess:
                # 找到文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名获得模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    # 运算出数据
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    print("After %s training stpe(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()

'''
After 29001 training stpe(s), validation accuracy = 0.9858
'''