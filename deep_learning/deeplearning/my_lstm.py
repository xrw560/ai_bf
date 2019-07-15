import tensorflow as tf


def _concat(x, output):
    return tf.concat([x, output], axis=1)


def _get_gate(x, size, name):
    with tf.variable_scope(name):
        x = _my_fc(x, size, name='w')
    x = tf.sigmoid(x)
    return x


class MyLSTM:
    scope_index = 0

    def __init__(self, state_size, output_size):
        self.output_size = output_size
        self.state_size = state_size
        self._reuse = None
        self._scope_name = "AaKie3845_%d" % MyLSTM.scope_index  # 每个LSTM的name是唯一的
        MyLSTM.scope_index += 1

    def zero_state(self, batch_size):
        return [[0.] * self.state_size for _ in range(batch_size)]

    def zero_output(self, batch_size):
        return [[0.] * self.output_size for _ in range(batch_size)]

    def __call__(self, x, output, state):
        with tf.variable_scope(self._scope_name, reuse=self._reuse):
            in_x = _concat(x, output)
            forget_gate = _get_gate(in_x, self.state_size, 'forget_gate')
            input_gate = _get_gate(in_x, self.output_size + x.shape[1].value, 'input_gate')
            output_gate = _get_gate(in_x, self.output_size, 'output_gate')

            state *= forget_gate

            in_x = tf.nn.tanh(in_x)
            in_x *= input_gate

            # we use concat instead of add(+) only because
            # the state_size is different with the output_size
            in_x = tf.concat([in_x, state], axis=1)

            # Please repeat the fc if you want add more hidden layers
            in_x = _my_fc(in_x, in_x.shape[1].value, name='in_x_fc')
            in_x = tf.nn.relu(in_x)

            state = _my_fc(in_x, self.state_size, name='state_fc')

            output = _my_fc(tf.nn.tanh(state), self.output_size, name='output_fc')
            # output = _my_fc(state, self.output_size, name='output_fc')
            output *= output_gate
            self._reuse = True
        return output, state


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
