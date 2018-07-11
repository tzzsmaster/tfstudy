# -*- coding: utf-8 -*-
"""
@author: tz_zs
(t3_1)
计算图的使用
"""
import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b

# a.graph查看张量所属的计算图，tf.get_default_graph()获取当前默认的计算图
print(a.graph is tf.get_default_graph())  # True

# tf.Graph()生成新的计算图，g1.as_default()设置为默认图
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", initializer=tf.zeros_initializer(), shape=[1])  # 定义变量“v”，并设置初始值为0

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer(), shape=[1])  # 定义变量“v”，并设置初始值为1

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))  # [ 0.]

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))  # [ 1.]


'''
《TensorFlow如何通过tf.device函数来指定运行每一个操作的设备？》 http://www.mamicode.com/info-detail-1987189.html


TensorFlow程序可以通过tf.device函数来指定运行每一个操作的设备。

这个设备可以是本地的CPU或者GPU，也可以是某一台远程的服务器。
TensorFlow会给每一个可用的设备一个名称，tf.device函数可以通过设备的名称，来指定执行运算的设备。比如CPU在TensorFlow中的名称为/cpu:0。

在默认情况下，即使机器有多个CPU，TensorFlow也不会区分它们，所有的CPU都使用/cpu:0作为名称。

–而一台机器上不同GPU的名称是不同的，第n个GPU在TensorFlow中的名称为/gpu:n。
–比如第一个GPU的名称为/gpu:0，第二个GPU名称为/gpu:1，以此类推。
–TensorFlow提供了一个快捷的方式，来查看运行每一个运算的设备。
–在生成会话时，可以通过设置log_device_placement参数来打印运行每一个运算的设备。

请看下面例子：

下面程序展示了log_device_placement参数的使用，在机器上直接运行代码：
–import tensorflowas tf
–a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
–b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
–c = a + b
–# 通过log_device_placement参数来输出运行每一个运算的设备。
–sess= tf.Session(config=tf.ConfigProto(log_device_placement=True))
–print sess.run(c)

在以上代码中，TensorFlow程序生成会话时加入了参数log_device_placement=True，所以程序会将运行每一个操作的设备输出到屏幕。

–除了可以看到最后的计算结果之外，还可以看到类似“add: /job:localhost/replica:0/task:0/cpu:0”这样的输出
–这些输出显示了执行每一个运算的设备。比如加法操作add是通过CPU来运行的，因为它的设备名称中包含了/cpu:0。
–在配置好GPU环境的TensorFlow中，如果操作没有明确地指定运行设备，那么TensorFlow会优先选择GPU。

在没有GPU的机器上运行，以上代码得到以下输出：
–Device mapping: no known devices.
–add: /job:localhost/replica:0/task:0/cpu:0
–b: /job:localhost/replica:0/task:0/cpu:0
–a: /job:localhost/replica:0/task:0/cpu:0
–[ 2. 4. 6.]
–‘‘‘


本文出自 “中科院计算所培训” 博客，谢绝转载！

TensorFlow如何通过tf.device函数来指定运行每一个操作的设备？

标签：tf.device函数   tensorflow   

原文地址：http://tcit1987.blog.51cto.com/8713743/1962455
'''

# tf.Graph().device函数来指定运行计算的设备
g = tf.Graph()
with g.device('/gpu:0'):
    result = a + b
'''
tf.device('/cpu:0')  是 Graph.device() 的包装，使用默认图
'''