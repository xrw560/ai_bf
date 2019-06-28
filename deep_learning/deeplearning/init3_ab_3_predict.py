# -*- conding:utf-8 -*-
from init3_ab_3_train import Tensors
import tensorflow as tf


def predict():
    tensors = Tensors()
    with tf.Session() as ss:
        # ss.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(ss, 'model/init3/mymodel')

        X = [[1, e] for e in range(1, 10)]
        Y = [[-2 * e[1] + 5] for e in X]
        print(X)
        print(Y)
        y_predict = ss.run(tensors.y_predict, feed_dict={
            tensors.x: X,
            # tensors.y: Y
        })
        print(y_predict)


if __name__ == '__main__':
    predict()
