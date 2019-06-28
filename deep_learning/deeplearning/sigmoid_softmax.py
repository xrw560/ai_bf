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


inputs = [2, 3, 5, 6]
print("Sigmoid Function Outputs: {}".format(sigmoid(inputs)))
print("Softmax Function Outputs: {}".format(softmax(inputs)))
