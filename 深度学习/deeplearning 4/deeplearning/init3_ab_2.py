import tensorflow as tf
import numpy as np


X = [e + np.random.random() for e in range(1, 10)]
Y = [-2*e+5+(np.random.random()/8-0.0625) for e in X]
X = [[1, e] for e in X]   # 9*2


def machine_learning():
    XX = np.mat(X)  # 9*2
    YY = np.mat(Y).T  # 9*1
    theta = (XX.T * XX).I * XX.T * YY
    return theta


class Tensors:
    def __init__(self, lr):
        ab = tf.get_variable('a', shape=[2,1])
        x = tf.placeholder(name='x', shape=[None, 2], dtype=tf.float32)
        y = tf.placeholder(name='y', shape=[None, 1], dtype=tf.float32)

        y_predict = tf.matmul(x, ab)
        loss = tf.reduce_mean((y_predict - y) ** 2)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train = optimizer.minimize(loss)

        self.train = train
        self.loss = loss
        self.ab = ab
        self.x = x
        self.y = y


def train():
    XX = np.mat(X)  # 9*2
    YY = np.mat(Y).T  # 9*1
    # XX = X  # 9*2
    # YY = [[e] for e in Y]  # 9 * 1
    print(XX)
    print(YY)
    tensors = Tensors(lr=0.001)
    with tf.Session() as ss:
        ss.run(tf.global_variables_initializer())
        for i in range(20000):
            _, loss = ss.run([tensors.train, tensors.loss], feed_dict={
                    tensors.x: XX,
                    tensors.y: YY
                })
            if i % 1000 == 0:
                print('Step %d, loss: %f' % (i, loss))
        print('final result ab :%s' % ss.run(tensors.ab))


if __name__ == '__main__':
    # print(machine_learning())
    train()

