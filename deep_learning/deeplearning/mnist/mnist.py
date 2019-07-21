# -*- coding: utf-8 -*-
from input_data import read_data_sets
import numpy as np
import tensorflow as tf

SAVE_PATH = "model/mnist"
SUMMARY_PATH = "summary"


def _fc(x, neurals, name='fc'):
    with tf.variable_scope(name):
        initializer = tf.initializers.random_normal(stddev=0.3)
        x_len = x.shape[1].value
        w = tf.get_variable('w', shape=[x_len, neurals], initializer=initializer)
        b = tf.get_variable('b', shape=[neurals], initializer=initializer)
        return tf.matmul(x, w) + b


class Tensors:
    def __init__(self, gpus):
        self.x_s = []
        self.y_s = []
        self.y_predict_s = []
        self.y_predict_digit_s = []
        self.precise_s = []
        self.optimizer_s = []
        self.gradient_s = []
        self.loss_s = []

        self.lr = tf.placeholder(tf.float32, name="lr")
        with tf.variable_scope("mnist"):
            for gpu_id in range(gpus):
                with tf.device('/gpu:%d' % gpu_id):
                    self._build_tensors()
                    tf.get_variable_scope().reuse_variables()

        self.loss = tf.reduce_mean(self.loss_s)
        self.precise = tf.reduce_mean(self.precise_s)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("precise", self.precise)
        self.summary = tf.summary.merge_all()
        self.minimize = self._get_minimize()

    def _build_tensors(self):
        x = tf.placeholder(tf.float32, shape=[None, 28 * 28])  # 向量
        t = tf.reshape(x, shape=[-1, 28, 28, 1])

        t = tf.layers.conv2d(t, 32, kernel_size=3, strides=1, padding='same', name="conv1")
        t = tf.layers.max_pooling2d(t, pool_size=2, strides=2)  # 14*14*32
        t = tf.nn.relu(t)

        t = tf.layers.conv2d(t, filters=64, kernel_size=3, strides=1, padding='same', name="conv2")
        t = tf.layers.max_pooling2d(t, pool_size=2, strides=2)  # 7*7*64
        t = tf.nn.relu(t)

        t = tf.reshape(t, shape=[-1, 7 * 7 * 64])
        t = _fc(t, 1000, name="fc1")
        t = tf.nn.relu(t)
        t = _fc(t, 10, name="fc2")

        y_predict = tf.nn.softmax(t)
        y = tf.placeholder(tf.int32, shape=[None])

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=t))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradient = optimizer.compute_gradients(loss)  # 梯度计算

        ## ------test--------##
        predict_digit = tf.argmax(y_predict, axis=1, output_type=tf.int32)
        precise = tf.reduce_mean(tf.cast(tf.equal(predict_digit, y), tf.float32))

        self.x_s.append(x)
        self.y_s.append(y)
        self.y_predict_s.append(y_predict)
        self.y_predict_digit_s.append(predict_digit)
        self.precise_s.append(precise)
        self.optimizer_s.append(optimizer)
        self.gradient_s.append(gradient)
        self.loss_s.append(loss)

    def _get_minimize(self):
        result = {}
        for gradient in self.gradient_s:  # 每个变量各一个梯度
            for grad, var in gradient:
                if not var in result:
                    result[var] = []
                result[var].append(grad)  # 同一个变量的梯度放一起
        result = [(tf.reduce_mean(result[var], axis=0), var) for var in result]
        return self.optimizer_s[0].apply_gradients(result)  # 变量共享，一个GPU上改变，其他的也会跟着改变


class Mnist:
    def __init__(self, gpus):
        self.gpus = gpus
        self.ds = read_data_sets("MNIST_data")
        graph = tf.Graph()
        with graph.as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            session = tf.Session(graph=graph, config=config)
            self.tensors = Tensors(self.gpus)
            self.session = session
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(session, SAVE_PATH)
                print("Restore model from %s successfully" % SAVE_PATH)
            except Exception as e:
                print(e)
                print("use a new model!!!")
                session.run(tf.global_variables_initializer())

    def train(self, lr=0.002, epoches=100, batch_size=12):
        samples_num = self.ds.train.num_examples
        steps = int(samples_num / batch_size / self.gpus)
        loss = 0.
        summary_writer = tf.summary.FileWriter(SUMMARY_PATH, graph=tf.get_default_graph())
        for i in range(epoches):
            for j in range(steps):
                feed_dict = {
                    self.tensors.lr: lr
                }
                for k in range(self.gpus):
                    samples, labels = self.ds.train.next_batch(batch_size)
                    feed_dict[self.tensors.x_s[k]] = samples
                    feed_dict[self.tensors.y_s[k]] = labels
                _, lo, summary = self.session.run([self.tensors.minimize, self.tensors.loss, self.tensors.summary],
                                                  feed_dict=feed_dict)
                loss += lo
                summary_writer.add_summary(summary)
                if j != 0 and j % 100 == 0:
                    print("Epoch: %d, step: %d, loss: %.5f, precise: %.5f" % (i, j, loss / 100, self.test(batch_size)),
                          flush=True)
                    loss = 0.
            self.saver.save(self.session, SAVE_PATH)
            print("Mode is saved into %s" % SAVE_PATH, flush=True)

    def test(self, batch_size=12):
        """测试"""
        samples_num = self.ds.test.num_examples
        # print("%d test samples are ready" % samples_num, flush=True)
        steps = int(samples_num / batch_size / self.gpus)

        precise = 0
        for j in range(steps):
            feed_dict = {}
            for k in range(self.gpus):
                samples, labels = self.ds.test.next_batch(batch_size)
                feed_dict[self.tensors.x_s[k]] = samples
                feed_dict[self.tensors.y_s[k]] = labels
            precise += self.session.run(self.tensors.precise, feed_dict=feed_dict)
        return precise / steps

    def predict(self, batch_size=20):
        samples_num = 50  # 取10个数据
        # print("%d test samples are ready" % samples_num, flush=True)
        steps = int(samples_num / batch_size / self.gpus)
        samples_s = []
        predict_s = []
        for j in range(steps):
            feed_dict = {}
            for k in range(self.gpus):
                samples, _ = self.ds.validation.next_batch(batch_size)
                feed_dict[self.tensors.x_s[k]] = samples
                samples_s.append(samples)
            predict_s += self.session.run(self.tensors.y_predict_digit_s, feed_dict=feed_dict)
        for samples, predicts in zip(samples_s, predict_s):
            for sample, predict in zip(samples, predicts):
                my_print(sample)
                print(predict)


def my_print(sample):
    sample = np.reshape(sample, [28, 28])
    for y in range(28):
        for x in range(28):
            print(" " if sample[y][x] == 0 else 'o', end='')
        print(flush=True)


if __name__ == "__main__":
    mnist = Mnist(gpus=2)
    # mnist.train(batch_size=50)
    # print(mnist.test())
    mnist.predict(batch_size=20)
