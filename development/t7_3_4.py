# -*- coding: utf-8 -*-
"""
@author: tz_zs
"""

import tensorflow as tf

from development.t7_2_2 import preprocess_for_train

# 创建 输入文件队列
files = tf.train.match_filenames_once("/path/to/file_pattern-*")
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 读、解析
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features={'image': tf.FixedLenFeature([], tf.string),
                                                                 'label': tf.FixedLenFeature([], tf.int64),
                                                                 'height': tf.FixedLenFeature([], tf.int64),
                                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                                 'channels': tf.FixedLenFeature([], tf.int64)})

image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

# 从原始图像的字符串数据解析出像素数组，并根据图像尺寸还原图像
decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])

# 神经网络输入层图片的大小
image_size = 299
# 前面的图像处理程序
distored_image = preprocess_for_train(decoded_image, image_size, image_size, None)

# 整理成batch
min_afteer_dequeue = 10000
batch_size = 100
capacity = min_afteer_dequeue + batch_size * 3
image_batch, label_batch = tf.train.shuffle_batch([distored_image, label], batch_size=batch_size, capacity=capacity,
                                                  min_after_dequeue=min_afteer_dequeue)

# 神经网络
...

with tf.Session() as sess:
    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        ...

    coord.request_stop()
    coord.join(threads)
