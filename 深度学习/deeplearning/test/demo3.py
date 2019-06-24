# -*- coding: utf-8 -*-

import tensorflow as tf

ss = tf.Session()
# 占位符,初值在run时给定
p = tf.placeholder(name="p", shape=[], dtype=tf.float32)

q = p * 3
print(ss.run(q, feed_dict={p: 300}))
print(ss.run(p, feed_dict={p: 400}))

p2 = tf.placeholder(shape=[], dtype=tf.float32)
q = p + p2
print(ss.run(q, feed_dict={p: 300, p2: 600}))
