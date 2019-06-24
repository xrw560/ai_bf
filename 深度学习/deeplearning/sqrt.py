import tensorflow as tf


# 张量中没有数据
class Tensors:
    def __init__(self, a, lr=0.01):
        x = tf.get_variable('x', dtype=tf.float32, initializer=0.1)
        loss = tf.square(a - x ** 2)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        self.train = opt.minimize(loss)  # 梯度下降
        self.x = x
        self.loss = loss


def train():
    a = 2
    ts = Tensors(a)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())  # 变量张量初始化
        for i in range(20000):
            session.run(ts.train)
            if i % 1000 == 0:
                loss = session.run(tf.sqrt(ts.loss))
                print('Train step %d, loss=%s' % (i, loss))
        print('sqrt(%s)=%s' % (a, session.run(ts.x)))


if __name__ == '__main__':
    train()
