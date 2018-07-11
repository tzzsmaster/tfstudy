# -*- coding: utf-8 -*-
"""
@author: tz_zs

完整的图片预处理样例
"""
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        # 随机亮度
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # 随机饱和度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # 随机色相
        image = tf.image.random_hue(image, max_delta=0.2)
        # 随机对比度
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, bbox):
    # 如果没有提供注释框，则关注整个图像
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])  # [[[ 0.  0.  1.  1.]]]

    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机截取图像
    # print(tf.shape(image).eval())  # [232 320   3]
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 大小
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))

    # 翻转
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # 随机色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))

    return distorted_image


image_raw_data = tf.gfile.FastGFile("picture.jpg", "rb").read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    for i in range(6):
        result = preprocess_for_train(img_data, 299, 299, boxes)
        plt.imshow(result.eval())
        plt.show()
