# -*- coding: utf-8 -*-
"""
@author: tz_zs

滑动平均模型
"""
import tensorflow as tf

# 定义一个变量，用于滑动平均计算
v1 = tf.Variable(0, dtype=tf.float32)
# 定义一个变量step,表示迭代的轮数，用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义滑动平均的对象
ema = tf.train.ExponentialMovingAverage(0.99, step)

# 定义执行保持滑动平均的操作,  参数为一个列表格式
maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    #  初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 通过ema.average(v1)获取滑动平均之后变量的取值，
    # print(sess.run(v1))  # 0.0
    # print(sess.run([ema.average_name(v1), ema.average(v1)]))  # [None, 0.0]
    print(sess.run([v1, ema.average(v1)]))  # [0.0, 0.0]

    # 更新变量v1的值为5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值，衰减率 min { decay , ( 1 + num_updates ) / ( 10 + num_updates ) }=0.1
    # 所以v1的滑动平均会被更新为 0.1*0 + 0.9*5 = 4.5
    sess.run(maintain_average_op)
    # print(sess.run(v1))  # 5.0
    # print(sess.run([ema.average_name(v1), ema.average(v1)]))  # [None, 4.5]
    print(sess.run([v1, ema.average(v1)]))  # [5.0, 4.5]

    # 更新step的值为10000。模拟迭代轮数
    sess.run(tf.assign(step, 10000))
    # 跟新v1的值为10
    sess.run(tf.assign(v1, 10))
    # 更新v1的滑动平均值。衰减率为 min { decay , ( 1 + num_updates ) / ( 10 + num_updates ) }得到 0.99
    # 所以v1的滑动平均值会被更新为 0.99*4.5 + 0.01*10 = 4.555
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))  # [10.0, 4.5549998]

    # 再次更新滑动平均值，将得到 0.99*4.555 + 0.01*10 =4.60945
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))  # [10.0, 4.6094499]
