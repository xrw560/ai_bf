import tensorflow as tf


class MyMultiLSTM:
    def __init__(self, lstm_list):
        self.lstm_list = lstm_list

    def __call__(self, x, output_list, state_list):
        result_outputs = []
        result_states = []
        for lstm, output, state in zip(self.lstm_list, output_list, state_list):
            output, state = lstm(x, output, state)
            result_outputs.append(output)
            result_states.append(state)
            x = output
        return result_outputs, result_states

    def zero_state(self, batch_size):
        result = []
        for lstm in self.lstm_list:
            result.append(lstm.zero_state(batch_size))
        return result

    def zero_output(self, batch_size):
        result = []
        for lstm in self.lstm_list:
            result.append(lstm.zero_output(batch_size))
        return result

