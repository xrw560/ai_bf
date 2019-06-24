import tensorflow as tf


class MyLSTM:

    def __init__(self, state_size, output_size):
        self.output_size = output_size
        self.state_size = state_size
        self._reuse = None

    def zero_state(self, batch_size):
        return [[0.] * self.state_size for _ in range(batch_size)]

    def zero_output(self, batch_size):
        return [[0.] * self.output_size for _ in range(batch_size)]

    def __call__(self, x, output, state):
        with tf.variable_scope("AaKie3845", reuse=self._reuse):
            in_x = tf.concat([x, output], axis=1)
            in_x = tf.concat([in_x, state], axis=1)

            in_x = _my_fc(in_x, in_x.shape[1].value, name='in_x_fc')
            in_x = tf.nn.relu(in_x)

            state = _my_fc(in_x, self.state_size, name='state_fc')
            output = _my_fc(in_x, self.output_size, name='output_fc')
            self._reuse = True
        return output, state


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


if __name__ == '__main__':
    lstm = MyLSTM()
    result = lstm(12, 34, 567)
    print(result)
