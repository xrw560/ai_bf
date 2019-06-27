# -*- coding: utf-8 -*-

import tensorflow as tf
from deep_learning.tensorflow_demo import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("data/", one_hot=True)
trainimgs, trainlabels, testimgs, testlabels = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
ntrain, ntest, dim, nclasses = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1], trainlabels.shape[1]
print(ntrain, ntest, dim, nclasses)
diminput = 28
dimhidden = 128
dimoutput = nclasses
nsteps = 28

weights = {
    'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])),
    'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))
}


def _RUN(_X, _W, _b, _nsteps, _name):
    # 1. Permute input from [batchsize,nsteps,diminput] -> [nsteps,batchsize,diminput]
    _X = tf.transpose(_X, [1, 0, 2])
    # 2. Reshape input to [nsteps*batchsize, diminput]
    _X = tf.reshape(_X, [-1, diminput])
    # 3.Input layer ->  hidden layer
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    # 4. Splite data to 'nsteps' chunks. An i-th chunck indicates i-th batch data
    _Hsplit = tf.split(_H, _nsteps, 0)
    # 5. Get LSTM's final output (_LSTM_0) and state (_LSTM_S)
    # both _LSTM_0 and _LSTM_S consist of 'batchsize' elements
    # only _LSTM_0 will be used to predict the output
    with tf.variable_scope(_name) as scope:
        scope.reuse_variables()
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)
        _LSTM_O, _LSTM_S = tf.nn.rnn(lstm_cell, _Hsplit, dtype=tf.float32)
    # 6. output
    _O = tf.matmul(_LSTM_O[-1], _W['out']) + _b['out']
    # return
    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit,
        'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O
    }


print("network ready")

learning_rate = 0.001
x = tf.placeholder(tf.float32, [None, nsteps, diminput])
y = tf.placeholder(tf.float32, [None, dimoutput])
myrun = _RUN(x, weights, biases, nsteps, 'basic')
pred = myrun['O']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))))
init = tf.global_variables_initializer()
print("network ready")

training_epochs = 5
batch_size = 16
display_step = 1
sess = tf.Session()
sess.run(init)
print("start optimization")
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    # loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
        # fit training using batch data
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        # compute average loss
        avg_cost += sess.run(cost, feed_dict=feeds) / total_batch
    # display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        print("Training accuracy: %.3f" % (train_acc))
        testimgs = testimgs.reshape((ntest, nsteps, diminput))
        feeds = {x: testimgs, y: testlabels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print("Test accuracy: %.3f" % (test_acc))
print("optimization finishedl")
