# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from deep_learning.tensorflow_demo import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# logistic regression model
actv = tf.nn.softmax(tf.matmul(x, W) + b)
# cost function
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))
# optimizer
learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# prediction
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
# accuracy
accr = tf.reduce_mean(tf.cast(pred, tf.float32))
# initializer
init = tf.global_variables_initializer()

training_epochs = 50
batch_size = 100
display_step = 5
# session
sess = tf.Session()
sess.run(init)
# mini-batch learning
for epoch in range(training_epochs):
    avg_cost = 0
    num_batch = int(mnist.train.num_examples / batch_size)
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds) / batch_size
    # display
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print("Epoch:%03d/%03d cost:%.f train_acc:%.3f test_acc:%.3f" % (
        epoch, training_epochs, avg_cost, train_acc, test_acc))
print("DONE")
