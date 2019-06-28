# -*- coding: utf-8 -*-
# import tensorflow as tf
# ss = tf.Session()

# x = tf.constant(30)
# y = tf.constant(40)
# z = x+y
# print(ss.run(z))

# x = tf.constant([1,2,3])
# # print(ss.run(x ** 2))
#
# y = tf.constant([4.0,5.0,6.0])
# print(ss.run(tf.exp(y)))
# print(ss.run(tf.log(y)))
#
# import math
#
# print(ss.run(tf.log(y) / math.log(2)))
# x = tf.cast(x,tf.float32)
# print(ss.run(x))

# ss.close()

y = lambda x, a: x ** 2 - a
y_x = lambda x: 2 * x
dx = lambda x, a, lr: -y(x, a) * y_x(x) * lr
x = 1
lr = 0.001
for _ in range(20000):
    x += dx(x, 2, lr)

y = lambda x, a: (x ** 2 - a) ** 2
y_x = lambda x, a: 2 * (x ** 2 - a) * 2 * x
dx = lambda x, a, lr: -y_x(x, a) * lr
for _ in range(20000):
    x += dx(x, 2, lr)
print(x)
