from input_data import read_data_sets
import cv2
import numpy as np
import tensorflow as tf

SAVA_PATH = "model_gan/gan"
SUMMARY_PATH = "summary_gan"


def _fc(x, neurals, name='fc'):
    with tf.variable_scope(name):
        initializer = tf.initializers.random_normal(stddev=0.3)
        x_len = x.shape[1].value
        w = tf.get_variable("w", shape=[x_len, neurals], initializer=initializer)
        b = tf.get_variable("b", shape=[neurals], initializer=initializer)
        return tf.matmul(x, w) + b


class Tensors:
    def __init__(self, gpus):
        self.gpus = gpus

        self.x_s = []
        self.x_y_s = []
        self.z_disc_y_s = []
        self.z_gen_y_s = []
        self.z_s = []

        self.x_loss_s = []
        self.x_optimizer_s = []
        self.x_gradient_s = []

        self.z_disc_loss_s = []
        self.z_disc_optimizer_s = []
        self.z_disc_gradient_s = []

        self.z_gen_loss_s = []
        self.z_gen_optimizer_s = []
        self.z_gen_gradient_s = []

        self.lr = tf.placeholder(tf.float32, name='lr')

        with tf.variable_scope("gan"):
            for gpu_id in range(self.gpus):
                with tf.device("/gpu:%d" % gpu_id):
                    self._build_tensors()
                    tf.get_variable_scope().reuse_variables()

        self.x_loss = tf.reduce_mean(self.x_loss_s)
        self.z_disc_loss = tf.reduce_mean(self.z_disc_loss_s)
        self.z_gen_loss = tf.reduce_mean(self.z_gen_loss_s)
        tf.summary.scalar("x_loss", self.x_loss)
        tf.summary.scalar("z_disc_loss", self.z_disc_loss)
        tf.summary.scalar("z_gen_loss", self.z_gen_loss)
        self.summary = tf.summary.merge_all()
        self.x_minimize = self._get_minimize(self.x_optimizer_s, self.x_gradient_s, 0)
        self.z_disc_minimize = self._get_minimize(self.z_disc_optimizer_s, self.z_disc_gradient_s, 1)
        self.z_gen_minimize = self._get_minimize(self.z_gen_optimizer_s, self.z_gen_gradient_s, 2)

    def _build_tensors(self):
        x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
        self.x_s.append(x)
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        with tf.variable_scope("discriminator"):
            x_result = self._get_discriminator(x)

        z = tf.placeholder(tf.float32, shape=[None, 100])
        self.z_s.append(z)
        with tf.variable_scope("generator"):
            fake = self._get_generator(z)
        with tf.variable_scope("discriminator", reuse=True):
            z_result = self._get_discriminator(fake)

        x_y = tf.placeholder(tf.float32, shape=[None, 2])
        self.x_y_s.append(x_y)
        x_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x_result, labels=x_y))
        x_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        var_list = [var for var in tf.trainable_variables() if var.name.startswith("gan/discriminator")]
        x_gradient = x_optimizer.minimize(x_loss, var_list=var_list)

        self.x_loss_s.append(x_loss)
        self.x_optimizer_s.append(x_optimizer)
        self.x_gradient_s.append(x_gradient)

        z_disc_y = tf.placeholder(tf.float32, shape=[None, 2])
        self.z_disc_y_s.append(z_disc_y)
        z_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z_result, labels=z_disc_y))
        z_disc_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        var_list = [var for var in tf.trainable_variables() if var.name.startswith("gan/discriminator")]
        z_disc_gradient = z_disc_optimizer.minimize(z_disc_loss, var_list=var_list)

        self.z_disc_loss_s.append(z_disc_loss)
        self.z_disc_optimizer_s.append(z_disc_optimizer)
        self.z_disc_gradient_s.append(z_disc_gradient)

        z_gen_y = tf.placeholder(tf.float32, shape=[None, 2])
        self.z_gen_y_s.append(z_gen_y)
        z_gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z_result, labels=z_gen_y))
        z_gen_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        var_list = z_gen_optimizer.minimize(z_gen_loss, var_list=var_list)
        z_gen_gradient = z_gen_optimizer.minimize(z_gen_loss, var_list=var_list)

        self.z_gen_loss_s.append(z_gen_loss)
        self.z_gen_optimizer_s.append(z_gen_optimizer)
        self.z_gen_gradient_s.append(z_gen_gradient)

    def _get_discriminator(self, x):
        t = tf.layers.conv2d(x, 32, 3, 1, padding='same', name='conv1')
        t = tf.layers.max_pooling2d(t, 2, 2)
        t = tf.nn.relu(t)
        t = tf.layers.conv2d(t, 64, kernel_size=3, strides=1, padding='same', name="conv2")
        t = tf.layers.max_pooling2d(t, pool_size=2, strides=2)
        t = tf.nn.relu(t)
        t = tf.reshape(t, shape=[-1, 7 * 7 * 64])
        t = _fc(t, 1000, name="fc1")
        t = tf.nn.relu(t)
        t = _fc(t, 2, name='fc2')
        return t

    def _get_generator(self, z):
        z = _fc(z, 1024, name='fc1')
        z = tf.reshape(z, [-1, 1, 1, 1024])

        z = tf.layers.conv2d_transpose(z, 512, 7, 1, name='conv1')
        z = tf.nn.relu(z)

        z = tf.layers.conv2d_transpose(z, 256, 5, 2, padding='same', name='conv2')
        z = tf.nn.relu(z)

        z = tf.layers.conv2d_transpose(z, 128, 5, 2, padding='same', name='conv3')
        z = tf.nn.relu(z)

        fake = tf.layers.conv2d_transpose(z, 1, 5, 1, padding='same', name='conv4')

        return fake

    def _get_minimize(self, optimizer_s, gradient_s, gpu_id):
        result = {}
        for gradient in gradient_s:
            for grad, var in gradient:
                if not var in result:
                    result[var] = []
                result[var].append(grad)
        result = [(tf.reduce_mean(result[var], axis=0)) for var in result]
        return optimizer_s[gpu_id % self.gpus].apply_gradients(result)


class Gan:
    def __init__(self, gpus):
        self.gpus = gpus
        self.ds = read_data_sets('MNIST_data')

        graph = tf.Graph()
        with graph.as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            session = tf.Session(graph=graph, config=config)
            self.tensors = Tensors(gpus)
            self.session = session
            self.saver = tf.train.Saver()
            try:
                self.saver.save(session, SAVA_PATH)
                print("Restore model from %s successfully" % SAVA_PATH)
            except Exception as e:
                print(e)
                print("Use a new model!!!")
                session.run(tf.global_variables_initializer())

    def train(self, lr=0.002, epoches=100, batch_size=4):
        sample_num = self.ds.tain.num_examples
        steps = int(sample_num / batch_size / self.gpus)
        summary_writer = tf.summary.FileWriter(SUMMARY_PATH, graph=tf.get_default_graph())
        for i in range(epoches):
            for j in range(steps):
                feed_dict = {
                    self.tensors.lr: lr
                }
                for k in range(self.gpus):
                    samples, _ = self.ds.train.next_batch(batch_size)
                    feed_dict[self.tensors.x_s[k]] = samples
                    feed_dict[self.tensors.x_y_s[k]] = [[1., 0.]] * batch_size
                    feed_dict[self.tensors.z_s[k]] = np.random.random([batch_size, 100])
                    feed_dict[self.tensors.z_disc_y_s[k]] = [[0., 1.]] * batch_size
                    feed_dict[self.tensors.z_gen_y_s[k]] = [[1., 0.]] * batch_size

                ts = self.tensors
                su, _, _, _ = self.session.run([ts.summary,
                                                ts.x_minimize, ts.z_disc_minimize, ts.z_gen_minimize],
                                               feed_dict=feed_dict)
                summary_writer.add_summary(su, i * steps + j)
                if j != 0 and j % 100 == 0:
                    print("Epoch: %d, step: %d" % (i, j), flush=True)
            self.saver.save(self.session, SAVA_PATH)
            print("model is saved into %s " % SAVA_PATH, flush=True)
