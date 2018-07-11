# -*- coding: utf-8 -*-
"""
@author: tz_zs

组合训练数据 batchingn
"""

import tensorflow as tf

# 正则匹配文件名
files = tf.train.match_filenames_once("/path/to/data.tfrecords-*")

# 创建输入队列
filename_queue = tf.train.string_input_producer(files, shuffle=False)
# print(filename_queue)  # <tensorflow.python.ops.data_flow_ops.FIFOQueue object at 0x00000196C9279AC8>

reader = tf.TFRecordReader()
_, serialized_example = reader.read(
    filename_queue)  # Tensor("ReaderReadV2:0", shape=(), dtype=string),Tensor("ReaderReadV2:1", shape=(), dtype=string)
features = tf.parse_single_example(serialized_example,
                                   features={'i': tf.FixedLenFeature([], tf.int64),
                                             'j': tf.FixedLenFeature([], tf.int64)})

example, label = features['i'], features['j']
# 一个 batch 中的样例个数
batch_size = 3
# 队列的最大容量
capactity = 1000 + 3 * batch_size

# 使用 tf.train.batch 组合样例 (tf.train.shuffle_batch)
example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capactity)

with tf.Session() as sess:
    tf.local_variables_initializer().run()
    # sess.run(files.initializer)
    print(sess.run(files))
    # [b'\\path\\to\\data.tfrecords-00000-of-00002'
    # b'\\path\\to\\data.tfrecords-00001-of-00002']

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)
    '''
    [0 0 1] [0 1 0]
    [1 0 0] [1 0 1]
    '''
    coord.request_stop()
    coord.join(threads)
