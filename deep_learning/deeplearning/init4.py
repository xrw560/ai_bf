import tensorflow as tf
import numpy as np


DIMS = 5
ALPHA = [0.83416808, 0.37521641, 0.27319377, 0.95008569, 0.2184194]
BETA  = 0.21266653994274987
THRESHHOLD = 1.5972234398894183

SAVE_PATH = 'model/init4/mymodel'

class Tensors:
    def __init__(self, lr=0.001):
        x = tf.placeholder(dtype=tf.float32, shape=[None, DIMS])
        w = tf.get_variable(name='w', shape=[DIMS, 1])
        b = tf.get_variable(name='b', shape=[1])
        y_predict = tf.matmul(x, w) + b     # important !!!!
        y_predict = tf.nn.tanh(y_predict)

        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        loss = tf.reduce_mean(tf.square(y - y_predict))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        minimize = optimizer.minimize(loss)

        self.x = x
        self.y = y
        self.y_predict = y_predict
        self.minimize = minimize
        self.loss = tf.sqrt(loss)


def train():
    tensors = Tensors()

    x = np.random.random([1000, DIMS])
    result = np.matmul(x, ALPHA) + BETA
    y = [[1 if e >= THRESHHOLD else -1] for e in result]

    with tf.Session() as ss:
        ss.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        try:
            saver.restore(ss, SAVE_PATH)
        except:
            print('use a new model')

        for i in range(5000):
            for j in range(2):
                _x = x[j * 100: j*500+500]
                _y = y[j * 100: j*500+500]
                _, loss = ss.run([tensors.minimize, tensors.loss],
                                 feed_dict={
                                     tensors.x: _x,
                                     tensors.y: _y
                                 })
            if i % 100 == 0:
                print('%d: loss = %s' % (i, loss))

        saver.save(ss, SAVE_PATH)


def predict():
    tensors = Tensors()

    x = np.random.random([1000, DIMS])
    result = np.matmul(x, ALPHA) + BETA
    y = [[1 if e >= THRESHHOLD else -1] for e in result]

    with tf.Session() as ss:
        saver = tf.train.Saver()
        saver.restore(ss, SAVE_PATH)
        y_predict = ss.run(tensors.y_predict, feed_dict={
            tensors.x: x
        })

    total = 0
    error = 0
    for xi, yi, yi_predict in zip(x, y, y_predict):
        yi = yi[0]
        yi_predict = yi_predict[0]
        print(xi, end='')
        print(' yi = %s, yi_predict = %s' % (yi, yi_predict))
        if yi > 0 and yi_predict < 0 or yi < 0 and yi_predict > 0:
            error += 1
        total += 1

    print('total = %s, error = %s' % (total, error))


if __name__ == '__main__':
    # train()
    predict()