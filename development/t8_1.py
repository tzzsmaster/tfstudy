# -*- coding: utf-8 -*-
"""
@author: tz_zs

最简单的 循环神经网络 前向传播过程
"""

import numpy as np

# 输入
X = [1, 2]
# 初始的状态
state = [0.0, 0.0]

# 循环体中 状态部分的权重、输入部分的权重、偏置
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# 输出的全连接
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 前向传播过程
for i in range(len(X)):
    # 循环体中的全连接神经网路
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)

    # 当前状态计算出的最终输出
    final_output = np.dot(state, w_output) + b_output

    print("before activation: ", before_activation)
    print("state: ", state)
    print("final_output: ", final_output)

'''
before activation:  [ 0.6  0.5]
state:  [ 0.53704957  0.46211716]
final_output:  [ 1.56128388]
before activation:  [ 1.2923401   1.39225678]
state:  [ 0.85973818  0.88366641]
final_output:  [ 2.72707101]
'''
