# -*- coding: utf-8 -*-
"""
@author: tz_zs

输入文件队列 写
"""
import tensorflow as tf


# TFRecord文件的格式
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 总共写入多少个文件
num_shards = 2
# 每个文件多少个数据
insatances_per_shard = 2

for i in range(num_shards):
    filename = ('/path/to/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    writer = tf.python_io.TFRecordWriter(filename)

    for j in range(insatances_per_shard):
        example = tf.train.Example(features=tf.train.Features(feature={'i': _int64_feature(i), 'j': _int64_feature(j)}))
        '''
        print(example)
        print(example.SerializeToString())
        
        features {
          feature {
            key: "i"
            value {
              int64_list {
                value: 0
              }
            }
          }
          feature {
            key: "j"
            value {
              int64_list {
                value: 0
              }
            }
          }
        }
        
        b'\n\x18\n\n\n\x01i\x12\x05\x1a\x03\n\x01\x00\n\n\n\x01j\x12\x05\x1a\x03\n\x01\x00'
        '''
        writer.write(example.SerializeToString())
    writer.close()
