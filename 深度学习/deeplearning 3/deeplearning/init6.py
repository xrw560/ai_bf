import tensorflow as tf
import numpy as np
import math

DIMS = 2
R1 = DIMS ** 0.5
R2 = (2**(1/DIMS)) * R1

SAVE_PATH = 'model/init6/mymodel'


class Tensors:
    def __init__(self, lr=0.001):
        x = tf.placeholder(dtype=tf.float32, shape=[None, DIMS])
        w = tf.get_variable(name='w', shape=[DIMS, 3], initializer=tf.initializers.random_normal())
        b = tf.get_variable(name='b', shape=[3], initializer=tf.initializers.random_normal())
        y_predict = tf.matmul(x, w) + b     # important !!!!
        y_predict = tf.nn.relu(y_predict)
        y_predict = tf.nn.softmax(y_predict)

        y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

        loss = tf.reduce_sum(-y * tf.log(y_predict+0.00000001), axis=1)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        minimize = optimizer.minimize(loss)

        self.x = x
        self.y = y
        self.y_predict = y_predict
        self.minimize = minimize
        self.loss = loss

        self.w = w
        self.b = b


def train():
    tensors = Tensors(lr=0.001)
    x, y = get_samples()

    with tf.Session() as ss:
        ss.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        try:
            saver.restore(ss, SAVE_PATH)
            print('restore model success!')
        except:
            print('use a new model!!!')

        batch_size = 500
        total = len(x)

        for i in range(5000):
            for j in range(int(total/batch_size)):
                _x = x[j * batch_size: (j + 1) * batch_size]
                _y = y[j * batch_size: (j + 1) * batch_size]
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

    x, y = get_samples(1000)

    with tf.Session() as ss:
        saver = tf.train.Saver()
        saver.restore(ss, SAVE_PATH)
        y_predict = ss.run(tensors.y_predict, feed_dict={
            tensors.x: x
        })

    total = 0
    error = 0
    for xi, yi, yi_predict in zip(x, y, y_predict):
        print(xi, end='')
        print(' yi = %s, yi_predict = %s' % (yi, yi_predict))
        y_max_i = np.argmax(yi)
        y_predict_max_i = np.argmax(yi_predict)
        if y_max_i != y_predict_max_i:
            error += 1
        total += 1

    print('total = %s, error = %s' % (total, error))


def get_samples(num=5000):
    x = np.random.random([num, DIMS]) * R2
    y = []
    n = [0] * 3
    for xi in x:
        yi = np.sqrt(np.sum(xi ** 2))
        if yi <= R1:
            yi = [1., 0., 0.]
            n[0] += 1
        elif yi <= R2:
            yi = [0., 1., 0.]
            n[1] += 1
        else:
            yi = [0., 0., 1.]
            n[2] += 1
        y.append(yi)

    print('num=', n)
    return x, y


if __name__ == '__main__':
    train()
    # predict()
    # x, y = get_samples(100)
    # print(x)
    # print(y)

