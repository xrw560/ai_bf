import tensorflow as tf
import numpy as np
import math
import threading

from PIL import Image,ImageDraw
from my_lstm import MyLSTM
from my_multi_lstm import MyMultiLSTM
from qts_util import read_poems, VOCAB_SIZE, chinese_to_index, index_to_chinese


STATE_SIZE = 200
OUTPUT_SIZE = 200
INPUT_SIZE = 200  # Should be equal to OUTPUT_SIZE
INPUT_REPEATS = 32  # m
OUTPUT_REPEATS = 32  # n
SAVE_PATH = 'model/init19/mymodel'
INIT_LR = 0.0002


def _my_fc(input, output_neurals, name):
    input_size = input.shape[1].value
    w = tf.get_variable(name=name + '_w',
                        initializer=tf.initializers.random_normal(stddev=0.1),
                        shape=[input_size, output_neurals])
    b = tf.get_variable(name=name + '_b',
                        initializer=tf.initializers.random_normal(stddev=0.1),
                        shape=[output_neurals]
                        )
    input = tf.matmul(input, w) + b
    return input


def word2vector(input, name):
    x = tf.one_hot(input, depth=VOCAB_SIZE)
    return _my_fc(x, OUTPUT_SIZE, name=name + '_fc')


class Tensors:
    def __init__(self, batch_size):
        encoder = MyLSTM(STATE_SIZE, INPUT_SIZE)  # MyMultiLSTM(..., ...)
        decoder = MyLSTM(STATE_SIZE, OUTPUT_SIZE)  # MyMultiLSTM(..., ...)

        state = encoder.zero_state(batch_size)
        output = encoder.zero_output(batch_size)
        inputs, outputs, todays = [], [], []
        os = []
        with tf.variable_scope('encoder'):
            for i in range(INPUT_REPEATS):
                input = tf.placeholder(dtype=tf.int32, shape=[None])
                inputs.append(input)
                input = word2vector(input, name='word2vector')
                output, state = encoder(input, output, state)
                os.append(output)
                tf.get_variable_scope().reuse_variables()

        os = tf.transpose(os, [1, 0, 2])
        v = tf.get_variable('attention', [INPUT_SIZE, OUTPUT_REPEATS])
        att = tf.matmul(os, [v] * batch_size)
        p = tf.nn.softmax(att, axis=1)
        i_s = tf.matmul(tf.transpose(p, [0, 2, 1]), os)
        i_s = tf.split(i_s, OUTPUT_REPEATS, axis=1)

        with tf.variable_scope('decoder'):
            output = decoder.zero_output(batch_size)

            for i in range(OUTPUT_REPEATS):
                today = tf.placeholder(dtype=tf.int32, shape=[None])
                todays.append(today)
                i_in = tf.reshape(i_s[i], [batch_size, -1])
                output, state = decoder(i_in, output, state)
                outputs.append(output)

        predicts = []
        with tf.variable_scope('loss'):
            sentence_loss = []
            for today, output in zip(todays, outputs):
                predict = _my_fc(output, VOCAB_SIZE, name='loss_fc')
                predicts.append(tf.argmax(tf.nn.softmax(predict), axis=1))
                single_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=today)

                # p1 = tf.one_hot(today, VOCAB_SIZE)
                # p2 = tf.nn.softmax(predict)
                # single_loss = tf.reduce_sum(-p1 * tf.log(p2), axis=1)

                sentence_loss.append(single_loss)
                tf.get_variable_scope().reuse_variables()

        loss = tf.reduce_mean(sentence_loss)
        tf.summary.scalar('my_rnn_loss', loss)

        lr = tf.get_variable(name='lr', shape=[], trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        minimize = optimizer.minimize(loss)

        self.x_inputs = inputs
        self.y_todays = todays
        self.y_predict = predicts
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
            self.session.run(self.assign, feed_dict={self.lr: INIT_LR})  # lr初始化

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
        samples = Samples()
        session = self.session
        batch_size = self.batch_size
        total = samples.num

        tensors = self.tensors
        file_writer=tf.summary.FileWriter('log19',graph=self.graph)
        step=0
        for i in range(epoches):
            for j in range(int(total/batch_size)):
                _x, _y = samples.get_samples(batch_size)
                feed_dict = {}
                temp = np.transpose(_x)   # _x.shape = [batch_size, 32]
                for x_input, x_value in zip(tensors.x_inputs, temp):
                    feed_dict[x_input] = [e for e in x_value]
                    # if len(x_value) != 100:
                    #     print("_x: %d * %d" % (len(_x), len(_x[0])))
                    #     print('x_value: %s' % (x_value), flush=True)
                    #     self.saver.save(session, SAVE_PATH)
                    #     return
                temp = np.transpose(_y)
                for y_today, y_value in zip(tensors.y_todays, temp):
                    feed_dict[y_today] = [e for e in y_value]
                _, loss,summary = session.run([tensors.minimize, tensors.loss,tensors.summary],
                                              feed_dict=feed_dict)
                # session.run(self.assign, feed_dict={self.lr: 0.0001})
                step+=1

                file_writer.add_summary(summary,step)
                # if loss < 0.4:
                #     lr = 0.0001
                # else:
                #     lr = 0.001
                # session.run(self.assign, feed_dict={self.lr: lr})

                print('%d: loss = %s, lr = %s' % (i, loss, session.run(tensors.lr)), flush=True)
            # if i != 0 and i % 10 == 0:
            # self.saver.save(session, SAVE_PATH)
            # print('Model saved!', flush=True)

        self.saver.save(session, SAVE_PATH)
        print('Finish!', flush=True)

    def predict(self, first_index):
        result = [first_index]
        tensors = self.tensors
        feed_dict = {
            tensors.x_inputs[0]: [first_index] * self.batch_size
        }

        for i in range(1, REPEATS):
            predicts = self.session.run(tensors.y_predict[i-1], feed_dict)
            result.append(predicts[0])
            feed_dict[tensors.x_inputs[i]] = predicts

        return result


def get_samples():
    x = read_poems()
    y = []
    for xi in x:
        yi = xi[1:]
        yi.append(10)
        y.append(yi)
    return x, y


class Samples:
    def __init__(self):
        self.x, self.y = get_samples()
        self.pos = 0

    @property
    def num(self):
        return len(self.x)

    def get_samples(self, batch_size):
        end = self.pos + batch_size
        if end <= len(self.x):
            x = self.x[self.pos: end]
            y = self.y[self.pos: end]
        else:
            x = self.x[self.pos:]
            y = self.y[self.pos:]
            end -= len(self.x)
            x += self.x[:end]
            y += self.y[:end]
        self.pos = end
        return x, y


class MyThread (threading.Thread):
    def __init__(self):
        super(MyThread, self).__init__()

    def run(self):
        with Init() as init:
            init.predict()


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
    # samples = get_samples()
    # print(samples[:10])

    init=Init(50)
    # first = chinese_to_index('海')[0]
    # result = init.predict(first)
    # poem = index_to_chinese(result)
    # print(poem)
    init.train(300)
    # init=Init(994)
    # init.predict()
