import tensorflow as tf
import numpy as np


DIMS = 5
ALPHA = [0.83416808, 0.37521641, 0.27319377, 0.95008569, 0.2184194]
BETA  = 0.21266653994274987
THRESHHOLD = 1.5972234398894183

SAVE_PATH = 'model/init5/mymodel'


class Tensors:
    def __init__(self, lr=0.001):
        x = tf.placeholder(dtype=tf.float32, shape=[None, DIMS])
        w = tf.get_variable(name='w', shape=[DIMS, 3])
        b = tf.get_variable(name='b', shape=[3])
        y_predict = tf.matmul(x, w) + b     # important !!!!
        y_predict = tf.nn.relu(y_predict)
        y_predict = tf.nn.softmax(y_predict)

        y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

        loss = tf.reduce_mean((y - y_predict)**2, axis=1)
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
    x = np.random.random([num, DIMS])
    result = np.matmul(x, ALPHA) + BETA
    y = [1 if e >= THRESHHOLD else -1 for e in result]
    y = [[1., 0., 0.] if yi < 0 else ([0., 1., 0.] if xi[0]+xi[1] < 1.0 else [0., 0., 1.]) for yi, xi in zip(y, x)]
    return x, y


if __name__ == '__main__':
    # train()
    predict()

