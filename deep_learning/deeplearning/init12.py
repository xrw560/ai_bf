# -*- conding:utf-8 -*-
"""RNN"""
import tensorflow as tf
import numpy as np
import math
import threading
from PIL import Image, ImageDraw

STATE_SIZE = 8  # 状态维度
REPEATS = 6
SAVE_PATH = 'model/init12/mymodel'


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

    return input


class Tensors:
    def __init__(self):
        state = tf.constant([0.] * STATE_SIZE)  # 初值为0
        inputs = []

        reuse = None  # tf.AUTO_REUSE不要随意用
        for i in range(REPEATS):
            input = tf.placeholder(dtype=tf.float32, shape=[None, 1])
            with tf.variable_scope('rnn1', reuse=reuse):  # 第一次新建，其他复用
                x = _my_fc(input, STATE_SIZE, 'fc1')  # 1维 -> 8维
                x = tf.nn.relu(x)
                state = state + x
                output = _my_fc(state, 1, 'out1')  # 8维转1维
                state = _my_fc(state, STATE_SIZE, 'state_change')
                state = tf.nn.relu(state)
                inputs.append(input)
                reuse = True

        today = tf.placeholder(dtype=tf.float32, shape=[None, 1])  # 今天实际值
        loss = tf.reduce_mean(tf.square(today - output))
        tf.summary.scalar('my_rnn_loss', loss)

        lr = tf.get_variable(name='lr', shape=[], trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        minimize = optimizer.minimize(loss)

        self.x_inputs = inputs
        self.y_today = today
        self.y_predict = output
        self.minimize = minimize
        self.loss = loss

        self.lr = lr
        self.summary = tf.summary.merge_all()


class Init:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tensors = Tensors()
            self.session = tf.Session(graph=self.graph)
            self.session.run(tf.global_variables_initializer())

            self.lr = tf.placeholder(tf.float32)
            self.assign = tf.assign(self.tensors.lr, self.lr)
            self.session.run(self.assign, feed_dict={self.lr: 0.001})  # lr初始化

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
        batch_size = 400
        total = len(x)

        tensors = self.tensors
        file_writer = tf.summary.FileWriter('log12', graph=self.graph)
        step = 0
        for i in range(epoches):
            for j in range(int(total / batch_size)):
                _x = x[j * batch_size: (j + 1) * batch_size]
                _y = y[j * batch_size: (j + 1) * batch_size]
                feed_dict = {tensors.y_today: [[e] for e in _y]}
                temp = np.transpose(_x)
                for x_input, x_value in zip(tensors.x_inputs, temp):
                    feed_dict[x_input] = [[e] for e in x_value]
                _, loss, summary = session.run([tensors.minimize, tensors.loss, tensors.summary],
                                               feed_dict=feed_dict)
                step += 1

                file_writer.add_summary(summary, step)
                if loss < 0.4:
                    lr = 0.0001
                else:
                    lr = 0.001
                session.run(self.assign, feed_dict={self.lr: lr})

            if i % 100 == 0:
                print('%d: loss = %s, lr = %s' % (i, loss, session.run(tensors.lr)))

        self.saver.save(session, SAVE_PATH)

    def predict(self):
        x, y = get_samples(6000)
        x, y = x[5000:], y[5000:]  # 前5000个为训练样本，预测时取5000~6000

        tensors = self.tensors
        ss = self.session

        feed_dict = {}
        temp = np.transpose(x)
        for x_input, x_value in zip(tensors.x_inputs, temp):
            feed_dict[x_input] = np.reshape(x_value, [-1, 1])
        y_predict = ss.run(tensors.y_predict, feed_dict=feed_dict)

        total = 0.
        error = 0.
        for xi, yi, yi_predict in zip(x, y, y_predict):
            error += abs(yi - yi_predict)
            total += yi
            print(' yi = %s, yi_predict = %s, error = %s' % (yi, yi_predict, error / total))

        print('total = %s, error = %s' % (total, error / total))


def get_samples(num=5000):
    x = []
    for _ in range(num):
        xi = _get_sample(x)
        x.append(xi)
    # x = [2/(1+math.exp(-e)) - 1 for e in x]
    x = [e / 7 for e in x]

    xx = []
    yy = []
    for i in range(len(x) - REPEATS):
        xx.append(x[i: i + REPEATS])
        yy.append(x[i + REPEATS])
    return xx, yy


class MyThread(threading.Thread):
    def __init__(self):
        super(MyThread, self).__init__()

    def run(self):
        with Init() as init:
            init.predict()


ALPHA = [123., 456., 789., 345]
BETA = 7


def _get_sample(x):
    """根据已有样本创建新样本
    第二个样本与第一个有关，第三个与第一二个有关，
    当样本数大于4个时，后面的取值与前四个有关"""
    randoms = max(0, 4 - len(x))  # 随机样本数
    previous = []
    for _ in range(randoms):
        previous.append(np.random.random() * 1000)

    for last in x[len(previous) - 4:]:
        previous.append(last)

    result = np.sum(np.array(previous) * ALPHA) % BETA
    return result


def do_test():
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


if __name__ == '__main__':
    # print(get_samples(10))
    np.random.seed(908523895)
    # print(get_samples(100))

    init = Init()
    init.train(1000)
    init.predict()

    pass
