import tensorflow as tf


class Tensors:
    def __init__(self, lr=0.01):
        x1 = tf.get_variable('x1',dtype=tf.float32, initializer=0.01)
        x2 = tf.get_variable('x2',dtype=tf.float32, initializer=0.01)
        loss = tf.square(x1 - 3) + tf.square(2 * x2 + 4)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        self.train = opt.minimize(loss)
        self.x1 = x1
        self.x2 = x2
        self.loss = loss


def train():
    ts = Tensors()
    with tf.Session() as ss:
        ss.run(tf.global_variables_initializer())
        for i in range(20000):
            ss.run(ts.train)
            if i % 1000 == 0:
                loss = ss.run(ts.loss)
                print('Step %d, loss: %d' % (i, loss))
        print('final result x1 :%d, x2:%d' % (ss.run(ts.x1), ss.run(ts.x2)))


if __name__ == '__main__':
    train()