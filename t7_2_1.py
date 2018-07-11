# -*- coding: utf-8 -*-
"""
@author: tz_zs
图像编码解码
"""
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("picture.jpg", 'rb').read()

with tf.Session() as sess:
    # 解码
    image_data = tf.image.decode_jpeg(image_raw_data)

    print(image_data.eval())
    # plt.imshow(image_data.eval())
    # plt.show()
    """
    [[[ 94 131  53]
      [ 87 125  48]
      [ 83 121  48]
      ..., 
      [ 29  63  13]
      [ 31  65  14]
      [ 34  69  15]]
    
     [[106 150  65]
      [100 143  61]
      [ 94 138  59]
      ..., 
      [ 30  64  13]
      [ 32  67  13]
      [ 34  69  15]]
    
     [[111 161  72]
      [103 153  66]
      [ 97 149  64]
      ..., 
      [ 33  65  15]
      [ 34  67  14]
      [ 35  68  13]]
    
     ..., 
     [[250 250 250]
      [250 250 250]
      [251 251 251]
      ..., 
      [ 13  26   8]
      [ 14  25   9]
      [ 12  23   9]]
    
     [[248 248 248]
      [249 249 249]
      [250 250 250]
      ..., 
      [ 14  29  10]
      [ 15  28  10]
      [ 14  27  10]]
    
     [[250 250 250]
      [251 251 251]
      [249 249 249]
      ..., 
      [ 16  31  10]
      [ 16  31  12]
      [ 16  29  11]]]
    """
    # image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)  # uint8→float32

    # 编码
    # encode_image = tf.image.encode_jpeg(image_data)# 接收的是 Tensor型uint8
    # with tf.gfile.GFile("output.jpg", 'wb') as f:
    #     f.write(encode_image.eval())

    image_data = tf.image.resize_images(image_data, [180, 267], method=1)
    # plt.imshow(image_data.eval())
    # plt.show()
    # print(tf.shape(image_data).eval())  # [180 267   3]

    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    '''
    print(boxes)
    Tensor("Const:0", shape=(1, 2, 4), dtype=float32)
    print(boxes.eval())
    [[[0.05        0.05        0.89999998  0.69999999]
      [0.34999999  0.47        0.5         0.56]]]
    '''
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(image_data), bounding_boxes=boxes)

    print(begin.eval())
    print(size.eval())
    print(bbox_for_draw.eval())
    '''
    [ 9 77  0]
    [69 56 -1]
    [[[ 0.03333334  0.23595506  0.94444442  0.97378278]]]
    '''
    tf.image.draw_bounding_boxes()

    image_data = tf.slice(image_data, begin, size)
    plt.imshow(image_data.eval())
    plt.show()
