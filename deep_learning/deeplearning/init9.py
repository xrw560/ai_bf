import tensorflow as tf
import numpy as np
import math
import threading

from PIL import Image,ImageDraw


SIZE=12

SAVE_PATH = 'model/init9/mymodel'


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
        x = tf.placeholder(dtype=tf.float32, shape=[None, SIZE,SIZE])
        self.x = x
        x=tf.reshape(x,shape=[-1,SIZE,SIZE,1])
        x=tf.layers.conv2d(x,16,(3,3),(1,1),padding='same')


        x=tf.layers.max_pooling2d(x,(2,2),(2,2),padding='same')

        x=tf.layers.conv2d(x,32,(3,3),padding='same')
        x=tf.layers.max_pooling2d(x,(2,2),(2,2),padding='same')


        # keep_prob=tf.placeholder(tf.float32)
        # x=tf.nn.dropout(x,keep_prob)

        x=tf.reshape(x,[-1,3*3*32])
        x=_my_fc(x,3,'fc')

        y_predict = tf.nn.softmax(x)



        y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

        loss = tf.reduce_sum(-y * tf.log(y_predict+0.00000001), axis=1)
        loss = tf.reduce_mean(loss)


        tf.summary.scalar('my_loss2', loss)



        lr = tf.get_variable(name='lr', shape=[], trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        minimize = optimizer.minimize(loss)


        self.y = y
        self.y_predict = y_predict
        self.minimize = minimize
        self.loss = loss

        self.lr = lr
        self.summary=tf.summary.merge_all()


class Init:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tensors = Tensors()
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
        batch_size = 200
        total = len(x)

        tensors = self.tensors
        file_writer=tf.summary.FileWriter('log2',graph=self.graph)
        step=0
        for i in range(epoches):
            for j in range(int(total/batch_size)):
                _x = x[j * batch_size: (j + 1) * batch_size]
                _y = y[j * batch_size: (j + 1) * batch_size]
                _, loss,summary = session.run([tensors.minimize, tensors.loss,tensors.summary],
                                 feed_dict={
                                     tensors.x: _x,
                                     tensors.y: _y
                                 })
                step+=1

                file_writer.add_summary(summary,step)
                if loss < 0.4:
                    lr = 0.0005
                else:
                    lr = 0.001
                session.run(self.assign, feed_dict={self.lr: lr})

            if i % 100 == 0:
                print('%d: loss = %s, lr = %s' % (i, loss, session.run(tensors.lr)))

        self.saver.save(session, SAVE_PATH)

    def predict(self):
        x, y = get_samples(1000)

        tensors = self.tensors
        ss = self.session
        y_predict = ss.run(tensors.y_predict, feed_dict={
            tensors.x: x
        })

        total = 0
        error = 0
        for xi, yi, yi_predict in zip(x, y, y_predict):
            print(xi, end='')
            print(' yi = %s, yi_predict = %s' % (yi, yi_predict))
            y_max_i = np.argmax(yi)
            y_predict_max_i = np.argmax(yi_predict)
            if y_max_i != y_predict_max_i:
                error += 1
            total += 1

        print('total = %s, error = %s' % (total, error))


def get_samples(num=5000):
    x,y=[],[]
    for _ in range(num):
        xi,yi=get_sample()
        xi=[[e/255.0 for e in row] for row in xi]

        x.append(xi)
        y.append(yi)
    return x, y

class MyThread (threading.Thread):
    def __init__(self):
        super(MyThread, self).__init__()

    def run(self):
        with Init() as init:
            init.predict()

def get_sample():


    img=Image.new('L', (SIZE, SIZE), 0)
    draw=ImageDraw.Draw(img)


    x=np.random.randint(0, int(SIZE / 2))
    y=np.random.randint(0, int(SIZE / 2))
    dx=np.random.randint(int(SIZE/2), SIZE - x)
    dy=np.random.randint(int(SIZE/2), SIZE - y)
    p=np.random.random()

    if p<0.333333:
        draw.ellipse((x,y,x+dx,y+dy),outline=255)
        y=[1.,0.,0.]

    elif p<0.666666:
        draw.line((x,y,x+dx,y+dy),fill=255)
        y=[0.,1.,0.]

    else:
        draw.rectangle((x, y, x + dx, y + dy), outline=255)
        y = [0., 0., 1.]

    ary=np.array(img)



    return ary,y



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

    init=Init()
    init.train(400)
    init.predict()
