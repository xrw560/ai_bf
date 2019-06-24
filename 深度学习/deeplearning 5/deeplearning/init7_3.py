import tensorflow as tf
import numpy as np
import math
import threading


DIMS = 2
R1 = DIMS ** 0.5
R2 = (2**(1/DIMS)) * R1

SAVE_PATH = 'model/init7_3/mymodel'


def _my_fc(input, output_neurals, name):
    input_size = input.shape[1].value
    w = tf.get_variable(name=name + '_w',
                        initializer=tf.initializers.random_normal(),
                        shape=[input_size, output_neurals])
    b = tf.get_variable(name=name + '_b',
                        initializer=tf.initializers.random_normal(),
                        shape=[output_neurals]
                        )
    input = tf.matmul(input, w) + b
    # return tf.nn.relu(input)
    return input


class Tensors:
    def __init__(self):
        x = tf.placeholder(dtype=tf.float32, shape=[None, DIMS])

        t = x
        t = _my_fc(t, 2, '1_x')
        t = _my_fc(t, 3, '2_x')
        # t = tf.nn.relu(t)

        y_predict = tf.nn.softmax(t)

        y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

        loss = tf.reduce_sum(-y * tf.log(y_predict+0.00000001), axis=1)
        loss = tf.reduce_mean(loss)

        lr = tf.get_variable(name='lr', shape=[], trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        minimize = optimizer.minimize(loss)

        self.x = x
        self.y = y
        self.y_predict = y_predict
        self.minimize = minimize
        self.loss = loss

        self.lr = lr


class Init:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tensors = Tensors()
            self.session = tf.Session(graph=self.graph)
            self.session.run(tf.global_variables_initializer())

            self.lr = tf.placeholder(tf.float32)
            self.assign = tf.assign(self.tensors.lr, self.lr)
            self.session.run(self.assign, feed_dict={self.lr: 0.001})

            try:
                self.saver = tf.train.Saver()
                self.saver.restore(self.session, SAVE_PATH)
                print('restore model success!')
            except:
                print('use a new model!!!')

    def __enter__(self):
        print('in Init.enter()')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('in Init.exit()')
        self.session.close()

    def train(self, epoches=5000):
        x, y = get_samples()
        session = self.session
        batch_size = 500
        total = len(x)

        tensors = self.tensors
        for i in range(epoches):
            for j in range(int(total/batch_size)):
                _x = x[j * batch_size: (j + 1) * batch_size]
                _y = y[j * batch_size: (j + 1) * batch_size]
                _, loss = session.run([tensors.minimize, tensors.loss],
                                 feed_dict={
                                     tensors.x: _x,
                                     tensors.y: _y
                                 })
                if loss < 0.4:
                    lr = 0.0001
                else:
                    lr = 0.001
                session.run(self.assign, feed_dict={self.lr: lr})

            if i % 100 == 0:
                print('%d: loss = %s, lr = %s' % (i, loss, session.run(tensors.lr)))

        self.saver.save(session, SAVE_PATH)

    def predict(self):
        x, y = get_samples(1000)

        tensors = self.tensors
        ss = self.session
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

    # def close(self):
    #     self.session.close()


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


class MyThread (threading.Thread):
    def __init__(self):
        super(MyThread, self).__init__()

    def run(self):
        with Init() as init:
            init.predict()


if __name__ == '__main__':
    with Init() as init:
        init.train(5000)

    th = []
    for _ in range(1):
        t = MyThread()
        th.append(t)
        t.start()

    for t in th:
        t.join()

    print('main thread is finished.')
