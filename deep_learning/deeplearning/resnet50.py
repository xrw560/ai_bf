# coding=utf8


import tensorflow as tf


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


class Tensors:
    def resnet_bottleneck_block(self, x, std_filters, resize=False):
        y = tf.layers.conv2d(x, std_filters, kernel_size=1, padding='same', strides=2 if resize else 1)
        y = tf.layers.batch_normalization(y, axis=0, training=self.training)
        y = tf.nn.relu(y)
        y = tf.layers.conv2d(y, std_filters, kernel_size=3, padding='same', )
        y = tf.layers.batch_normalization(y, axis=0, training=self.training)
        y = tf.nn.relu(y)
        y = tf.layers.conv2d(y, 4 * std_filters, kernel_size=1, padding='same', )
        y = tf.layers.batch_normalization(y, axis=0, training=self.training)

        if x.shape[-1].value != 4 * std_filters:
            x = tf.layers.conv2d(x, 4 * std_filters, kernel_size=1, padding='same')

        x += y
        x = tf.nn.relu(x)
        return x

    def __init__(self):
        x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x')
        self.x = x
        x = tf.layers.conv2d(x, 64, kernel_size=7, strides=2, padding='same')  # ==> 112*112*64
        self.training = tf.placeholder(tf.bool, name='training')
        x = tf.layers.batch_normalization(x, axis=0, training=self.training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(pool_size=3, strides=2, padding='same')     # ==> 56*56*64

        x = self.resnet_bottleneck_block(x, 64, False)
        x = self.resnet_bottleneck_block(x, 64, False)
        x = self.resnet_bottleneck_block(x, 64, False)

        x = self.resnet_bottleneck_block(x, 128, True)      # ==> 28 * 28 * 512
        x = self.resnet_bottleneck_block(x, 128, False)
        x = self.resnet_bottleneck_block(x, 128, False)
        x = self.resnet_bottleneck_block(x, 128, False)

        x = self.resnet_bottleneck_block(x, 256, True)      # ==> 14 * 14 * 1024
        x = self.resnet_bottleneck_block(x, 256, False)
        x = self.resnet_bottleneck_block(x, 256, False)
        x = self.resnet_bottleneck_block(x, 256, False)
        x = self.resnet_bottleneck_block(x, 256, False)
        x = self.resnet_bottleneck_block(x, 256, False)

        x = self.resnet_bottleneck_block(x, 512, True)      # ==> 7 * 7 * 2048
        x = self.resnet_bottleneck_block(x, 512, False)
        x = self.resnet_bottleneck_block(x, 512, False)

        x = tf.layers.average_pooling2d(x, 7, 1)            # ==> 1 * 1 * 2048
        x = tf.reshape(x, [-1, 2048])
        x = _my_fc(x, 1000)

        y = tf.placeholder(tf.int32, shape=[None])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
        lr = tf.placeholder(dtype=tf.float32, name='lr')
        optimizer = tf.train.AdamOptimizer(lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            minimize = optimizer.minimize(loss)

        self.minimize = minimize
        self.y = y
        self.loss = loss
        self.lr = lr


if __name__ == '__main__':
    import numpy as np
    # normalize the 100 samples along with each dimention

    # data = np.random.randint(-3, 3, [100, 8])

    # data = tf.placeholder(tf.float32, shape=[None, 8])
    # data_avg = tf.reduce_mean(data, axis=0)
    # avg = tf.get_variable('avg', trainable=False, shape=[8])
    # if in_traing:
    #    with tf.control_dependencies([tf.assign(avg, p * avg + (1 - p) * data_avg)]):
    #        y = (data - avg) / std
    # else:
    #     y = (data - avg) / std

    # avg = np.mean(data, axis=0)
    # std = np.sqrt(np.mean(data**2, axis=0) - avg**2)
    # print(data)
    # print(avg)
    # print(std)
    # b = (data - avg)/std
    # print(b)
    #
    # print(np.sum(b, axis=0))      # expect to get eight 0's.
    # print(np.sum(b**2, axis=0))   # expect to get eight 100's.



    # a = tf.constant(30)
    # x = tf.get_variable('x', initializer=10)
    # with tf.control_dependencies([tf.assign(x, 999)]):
    #     y = 3 * a
    #
    # with tf.Session() as ss:
    #     ss.run(tf.global_variables_initializer())
    #     print(ss.run([x,y]))
