# -*- coding: utf-8 -*-
"""
@author: tz_zs(t3_2)
张量  三个属性：name、shape、type
"""
import tensorflow as tf

# tf.constant是一个计算，正计算的结果为一个张量，保存在变量a中
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = tf.add(a, b, name="add")  # 与“result = a + b”的区别在于，“tf.add”可指定“name”，而+得到的张量的默认名称为“add”
print(a)  # Tensor("a:0", shape=(2,), dtype=float32)
print(b)  # Tensor("b:0", shape=(2,), dtype=float32)
print(result)  # Tensor("add:0", shape=(2,), dtype=float32)

'''
#类型（type）需要匹配
c = tf.constant([2, 3])
print(c)  # Tensor("Const:0", shape=(2,), dtype=int32)
result2 = a + c
# ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32: 'Tensor("Const:0", shape=(2,), dtype=int32)'
'''

'''
# 维度（shape）必须一致
d = tf.constant([2.0, 3.0, 4.0])
print(d)
result3 = a + d
# ValueError: Dimensions must be equal, but are 2 and 3 for 'add_1' (op: 'Add') with input shapes: [2], [3].
'''
