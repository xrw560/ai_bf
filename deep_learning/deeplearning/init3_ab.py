import tensorflow as tf
import numpy as np

X = [e + np.random.random() for e in range(1, 10)]
Y = [2 * e + 1 + (np.random.random() / 8 - 0.0625) for e in X]
X = [[1, e] for e in X]  # 9*2


def machine_learning():
    XX = np.mat(X)  # 9*2
    YY = np.mat(Y).T  # 9*1
    theta = (XX.T * XX).I * XX.T * YY
    return theta


class Tensors:
    def __init__(self, lr):
        a = tf.get_variable('a', initializer=0.1)
        b = tf.get_variable('b', initializer=0.1)
        s = 0
        for ((x0, x1), y) in zip(X, Y):
            y_predict = x0 * b + x1 * a
            s += (y_predict - y) ** 2
        loss = s / len(X)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        _train = optimizer.minimize(loss)

        self.train = _train
        self.loss = loss
        self.a = a
        self.b = b


def train():
    tensors = Tensors(lr=0.001)
    with tf.Session() as ss:
        ss.run(tf.global_variables_initializer())
        for i in range(20000):
            ss.run(tensors.train)
            if i % 1000 == 0:
                loss = ss.run(tensors.loss)
                print('Step %d, loss: %f' % (i, loss))
        print('final result a :%f, b:%f' % (ss.run(tensors.a), ss.run(tensors.b)))


if __name__ == '__main__':
    # print(machine_learning())
    train()
