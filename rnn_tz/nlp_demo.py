#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

"""
@author:    zmate 
            Jiang Ningwang

@time:      18-7-5 下午5:51
"""

import numpy as np
import tensorflow as tf
from tutorials.rnn.ptb import reader

DATA_PATH = "source_data/data"
HIDDEN_SIZE = 200

NUM_LAYERS = 2
VOCAB_SIZE = 10000

LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 2
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5


# 通过一个 PTBModel 类来描述模型，方便维护神经网络中的状态
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self._batch_size = batch_size
        self._num_steps = num_steps

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # LSTM
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        # 初始化
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # 单词向量
        embedding = tf.get_variable(name="embedding", shape=[VOCAB_SIZE, HIDDEN_SIZE])

        # 转化
        inputs = tf.nn.embedding_lookup(params=embedding, ids=self.input_data)

        # dropout
        if is_training: inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 输出
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tensor=tf.concat(axis=1, values=outputs), shape=[-1, HIDDEN_SIZE])

        # 全链接
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        #损失
        # tf.nn.seq2seq.sequence_loss_by_example

        # TODO


def run_epoch(session, model, data, train_op, output_log):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    # TODO
    return


def main():
    # 获取原始数据
    train_data, valid_data, test_data, _ = reader.ptb_raw_data("source_data/data/")

    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

        # with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        #     eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

        # with tf.Session() as session:
        #     tf.initialize_all_variables().run()
        #     # 训练
        #     for i in range(NUM_EPOCH):
        #         print("In iteration: %d" % (i + 1))
        #         run_epoch(session, train_model, train_data, train_model.train_op, True)

        #     # 验证
        #     valid_perplexity = run_epoch(session, eval_model, valid_data, tf.no_op(), False)
        #     print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, float(valid_perplexity)))
        #
        # # 测试
        # test_perplexity = run_epoch(session, eval_model, test_data, tf.no_op(), False)
        # print("Test Perplexity: %.3f" % float(test_perplexity))


if __name__ == '__main__':
    main()
