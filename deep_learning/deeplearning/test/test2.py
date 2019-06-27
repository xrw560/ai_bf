# -*- coding: utf-8 -*-

import tensorflow as tf

ss = tf.Session()
x = tf.constant([[1.,2.,3.],[10.,11.,12.]])
y = tf.constant([[0.,-1.,-2.],[-4.,-5.,-6.]])

# print(x.shape)
# print(ss.run(tf.shape(x)))

z = tf.transpose(x)
print(ss.run(z))
print(z.shape)

p = tf.matmul(y,z)
print(z.shape)
print(ss.run(p))

ss.close()