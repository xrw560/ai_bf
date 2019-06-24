import tensorflow as tf
import numpy as np
import math
import threading

from PIL import Image,ImageDraw
from my_lstm import MyLSTM
from my_multi_lstm import MyMultiLSTM


STATE_SIZE = 200
OUTPUT_SIZE = 200
REPEATS = 6
SAVE_PATH = 'model/qts/mymodel'

VOCAB_SIZE = 8000

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


def word2vector(input, name):
    x = tf.one_hot(input, depth=VOCAB_SIZE)
    return _my_fc(x, OUTPUT_SIZE, name=name + '_fc')


class Tensors:
    def __init__(self, batch_size):
        multi_lstm = MyMultiLSTM([MyLSTM(STATE_SIZE, OUTPUT_SIZE), MyLSTM(STATE_SIZE, OUTPUT_SIZE)])

        state = multi_lstm.zero_state(batch_size)
        output = multi_lstm.zero_output(batch_size)
        inputs, outputs, todays = [], [], []

        with tf.variable_scope('qts'):
            for i in range(REPEATS):
                input = tf.placeholder(dtype=tf.int32, shape=[None])
                inputs.append(input)

                input = word2vector(input, name='word2vector')
                output, state = multi_lstm(input, output, state)
                outputs.append(output[-1])
                today = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SIZE])
                todays.append(today)
                tf.get_variable_scope().reuse_variables()

        loss = 0
        for today, output in zip(todays, outputs):
            loss += tf.square(today - output)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('my_rnn_loss', loss)

        lr = tf.get_variable(name='lr', shape=[], trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        minimize = optimizer.minimize(loss)

        self.x_inputs = inputs
        self.y_todays = todays
        self.y_predict = outputs
        self.minimize = minimize
        self.loss = loss

        self.lr = lr
        self.summary=tf.summary.merge_all()


class Init:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tensors = Tensors(batch_size)
            self.session = tf.Session(graph=self.graph)
            self.session.run(tf.global_variables_initializer())

            self.lr = tf.placeholder(tf.float32)
            self.assign = tf.assign(self.tensors.lr, self.lr)
            self.session.run(self.assign, feed_dict={self.lr: 0.001})#lr初始化

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
        batch_size = self.batch_size
        total = len(x)

        tensors = self.tensors
        file_writer=tf.summary.FileWriter('logqts',graph=self.graph)
        step=0
        for i in range(epoches):
            for j in range(int(total/batch_size)):
                _x = x[j * batch_size: (j + 1) * batch_size]
                _y = y[j * batch_size: (j + 1) * batch_size]
                feed_dict = {}
                temp = np.transpose(_x)
                for x_input, x_value in zip(tensors.x_inputs, temp):
                    feed_dict[x_input] = [[e] for e in x_value]
                temp = np.transpose(_y)
                for y_today, y_value in zip(tensors.y_todays, temp):
                    feed_dict[y_today] = [[e] for e in y_value]
                _, loss,summary = session.run([tensors.minimize, tensors.loss,tensors.summary],
                                              feed_dict=feed_dict)
                step+=1

                file_writer.add_summary(summary,step)
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
        x, y = x[5000:], y[5000:]

        tensors = self.tensors
        ss = self.session

        feed_dict = {}
        temp = np.transpose(x)
        for x_input, x_value in zip(tensors.x_inputs, temp):
            feed_dict[x_input] = np.reshape(x_value, [-1, 1])
        y_predict = ss.run(tensors.y_predict[-1], feed_dict=feed_dict)

        total = 0.
        error = 0.
        for xi, yi, yi_predict in zip(x, y, y_predict):
            error += abs(yi - yi_predict)
            total += yi
            print(' yi = %s, yi_predict = %s, error = %s' % (yi, yi_predict, error/total))

        print('total = %s, error = %s' % (total, error/total))


def get_samples(num=5000):
    x = []
    for _ in range(num):
        xi = _get_sample(x)
        x.append(xi)
    # x = [2/(1+math.exp(-e)) - 1 for e in x]
    x = [e/7 for e in x]

    xx = []
    yy = []
    for i in range(len(x) - REPEATS):
        xx.append(x[i: i + REPEATS])
        yy.append(x[i+1: i + 1 + REPEATS])
    return xx, yy


class MyThread (threading.Thread):
    def __init__(self):
        super(MyThread, self).__init__()

    def run(self):
        with Init() as init:
            init.predict()


ALPHA =[123., 456., 789., 345]
BETA = 7


def _get_sample(x):
    randoms = max(0, 4 - len(x))
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
    init=Init(400)
    # init.train(1000)
    # init=Init(994)
    # init.predict()
