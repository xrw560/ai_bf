# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(inputs):
    """
    Calculate the sigmoid for the give inputs (array)
    :param inputs:
    :return:
    """
    sigmoid_scores = [1 / float(1 + np.exp(-x)) for x in inputs]
    return sigmoid_scores


def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))


if __name__ == "__main__":
    # inputs = [2, 3, 5, 6]
    # print("Sigmoid Function Outputs: {}".format(sigmoid(inputs)))
    # print("Softmax Function Outputs: {}".format(softmax(inputs)))
    import tensorflow as tf

    a = tf.constant([[1., 2., 3.], [3, 2, 1]])
    b = tf.nn.softmax(a)
    with tf.Session() as ss:
        c = ss.run(b)
    print(c)
