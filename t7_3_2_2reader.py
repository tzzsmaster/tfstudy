# -*- coding: utf-8 -*-
"""
@author: tz_zs

输入文件队列 读
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

with tf.Session() as sess:
    tf.local_variables_initializer().run()
    # sess.run(files.initializer)
    print(sess.run(files))
    # [b'\\path\\to\\data.tfrecords-00000-of-00002'
    # b'\\path\\to\\data.tfrecords-00001-of-00002']

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    '''
    [0, 0]
    [0, 1]
    [1, 0]
    [1, 1]
    [0, 0]
    [0, 1]
    '''
    coord.request_stop()
    coord.join(threads)
