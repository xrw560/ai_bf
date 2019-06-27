# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from deep_learning.tensorflow_demo import input_data

print("Download and Extract MNIST dataset")
mnist = input_data.read_data_sets("data/", one_hot=True)
print("type of 'mnist' is %s " % (type(mnist)))
print("number of train data is %d " % (mnist.train.num_examples))
print("number of test data is %d " % (mnist.test.num_examples))

# What does the data of MNIST look like?
print("What does the data of MNIST look like?")
trainimg = mnist.train.images
trainlable = mnist.train.labels
testimg = mnist.test.images
testlable = mnist.test.labels
print("type of trainimg is %s" % (type(trainimg)))
print("type of tainlabel is %s" % (type(trainlable)))
print("type of testimg is %s" % (type(testimg)))
print("type of testlabel is %s" % (type(testlable)))

print("shape of trainimg is %s" % (trainimg.shape,))
print("shape of tainlabel is %s" % (trainlable.shape,))
print("shape of testimg is %s" % (testimg.shape,))
print("shape of testlabel is %s" % (testlable.shape,))

print("How does the training data look like?")
nsample = 5
randidx = np.random.randint(trainimg.shape[0], size=nsample)
for i in randidx:
    curr_img = np.reshape(trainimg[i, :], (28, 28))
    curr_label = np.argmax(trainlable[i, :])  # label
    plt.matshow(curr_img, cmap=plt.get_cmap("gray"))
    plt.title("" + str(i) + "th Training Data Label is " + str(curr_label))
    print("" + str(i) + "th Training Data Label is " + str(curr_label))
    plt.savefig("fig/" + str(i) + '.png', bbox_inches='tight')
# plt.show()
