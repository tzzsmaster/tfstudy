"""
@author: tz_zs(t3_3)

关于初始化、张量的计算取值等
"""

import tensorflow as tf

print("探究初始化的过程")
v1 = tf.Variable([1.0, 2.0])
v2 = tf.Variable([2.0, 3.0])
print(v1)  # <tf.Variable 'Variable:0' shape=(2,) dtype=float32_ref>
print(v2)  # <tf.Variable 'Variable_1:0' shape=(2,) dtype=float32_ref>
v3 = v1 + v2
print(v3)  # Tensor("add:0", shape=(2,), dtype=float32)

# 下面这种初始化方式会报错,因为没有指定默认会话
# ValueError: Cannot execute operation using `run()`: No default session is registered. Use `with sess.as_default():`
# or pass an explicit session to `run(session=sess)`
# sess = tf.Session()
# tf.global_variables_initializer().run()


# 下面这些是可以的
# 1、下面这种直接使用会话sess去run的
print("#" * 20, "1")
sess1 = tf.Session()
sess1.run(tf.global_variables_initializer())
print(sess1.run(v3))  # [ 3.  5.]
print(v3.eval(session=sess1))  # [ 3.  5.]

# 2、下面这种手动指定了默认的sess
print("#" * 20, "2")
sess2 = tf.Session()
with sess2.as_default():
    tf.global_variables_initializer().run()
    print(v3.eval())  # [ 3.  5.]

# 3、下面这种生成了上下文来管理sess，sess关心整个with内的结果
print("#" * 20, "3")
with tf.Session() as sess3:
    tf.global_variables_initializer().run()
    print(sess3.run(v3))  # [ 3.  5.]
    print(v3.eval(session=sess3))  # [ 3.  5.] 多此一举
    print(v3.eval())  # [ 3.  5.]

#################################################################
# tf.global_variables_initializer()的结构
# name: "init"
# # op: "NoOp"
# input: "^Variable/Assign"
# input: "^Variable_1/Assign"

with tf.Session() as sess4:
    sess4.run(tf.global_variables_initializer())
    print(sess4.run(v3))  # [ 3.  5.]
    print(v3.eval(session=sess4))  # [ 3.  5.]
    print(v3.eval())  # [ 3.  5.]

#################################################################
print("#######" * 10)
print("探究张量的值的计算")
#################################################################

v4 = tf.constant([1.0, 2.0])
v5 = tf.constant([2.0, 3.0])
v6 = v4 + v5
print(v4)  # Tensor("Const:0", shape=(2,), dtype=float32)
print(v5)  # Tensor("Const_1:0", shape=(2,), dtype=float32)
print(v6)  # Tensor("add:0", shape=(2,), dtype=float32)

# 第一种,手动指定了默认sess
sess5 = tf.Session()
with sess5.as_default():
    print(sess5.run(v6))  # [ 3.  5.]
    print(v6.eval(session=sess5))  # [ 3.  5.]
    print(v6.eval())  # [ 3.  5.]

# 第二种，还是生成的上下文管理的
with tf.Session() as sess6:
    print(sess6.run(v6))  # [ 3.  5.]
    print(v6.eval(session=sess6))  # [ 3.  5.]
    print(v6.eval())  # [ 3.  5.]

# 第三种，主动运行或指定会话
sess7 = tf.Session()
print(sess7.run(v6))  # [ 3.  5.]
print(v6.eval(session=sess7))  # [ 3.  5.]
# 下面这个会报错，因为没有默认会话____ValueError: Cannot evaluate tensor using `eval()`:
# No default session is registered. Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`
# print(v6.eval())
