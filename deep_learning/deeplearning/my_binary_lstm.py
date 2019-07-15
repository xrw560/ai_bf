# -*- conding:utf-8 -*-
"""
双向LSTM
"""
import tensorflow as tf


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


class MyBinaryLSTM:
    def __init__(self, lstm1, lstm2, batch_size, num_steps):
        self.lstm1 = lstm1
        self.lstm2 = lstm2
        self.num_steps = num_steps
        self.batch_size = batch_size

        inputs, outputs = [], []

        for _ in range(self.num_steps):
            inputs.append(tf.placeholder(tf.float32, shape=[None, lstm1.output_size]))
            outputs.append(tf.placeholder(tf.float32, shape=[None, lstm2.output_size]))

        state1 = self.lstm1.zero_state(batch_size)
        state2 = self.lstm2.zero_state(batch_size)
        output1 = self.lstm1.zero_output(batch_size)
        output2 = self.lstm2.zero_output(batch_size)

        predict1_s, predict2_s = [], []

        for i in range(self.num_steps):
            input1 = inputs[i]
            input2 = inputs[self.num_steps - 1 - i]
            output1, state1 = self.lstm1(input1, output1, state1)
            output2, state2 = self.lstm2(input2, output2, state2)
            predict1_s.append(output1)
            predict2_s.insert(0, output2)

        predicts = []
        for predict1, predict2, i in zip(predict1_s, predict2_s, range(num_steps)):
            predict = tf.concat([predict1, predict2], axis=1)
            predict = _my_fc(predict, self.lstm1.output_size, name="fc_%d" % i)
            predicts.append(predict)

        self.inputs = inputs
        self.outputs = outputs
        self.predicts = predicts
