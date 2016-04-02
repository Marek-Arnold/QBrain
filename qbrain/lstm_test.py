__author__ = 'Marek'
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell
import random


class NumberCounter:
    EMPTY_NUM = [0, 0]
    ONE_NUM = [1, 0]
    TWO_NUM = [0, 1]
    NUMS = [EMPTY_NUM, ONE_NUM, TWO_NUM]

    def __init__(self, seq_width=2):
        self.lstm_size = 2
        self.seq_width = seq_width
        initializer = tf.random_uniform_initializer(-1, 1)

        self.seq_input = tf.placeholder(tf.float32, [1, self.seq_width])
        self.expected_output = tf.placeholder(tf.float32, [1, self.seq_width])
        self.expected_output_valid = tf.placeholder(tf.float32, [1])

        self.cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size, self.seq_width, initializer=initializer)
        self.initial_state = self.cell.zero_state(1, tf.float32)

        # ========= This is the most important part ==========
        # output will be of length 4 and 6
        # the state is the final state at termination (stopped at step 4 and 6)

        self.state = self.initial_state
        self.outputs, self.state = tf.nn.rnn(self.cell, [self.seq_input], initial_state=self.state)

        self.loss = tf.mul(tf.reduce_sum(tf.pow(tf.sub(self.outputs[0], self.expected_output), 2)), self.expected_output_valid)

        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
        # usual crap
        iop = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(iop)

    # batch_series_input: n_steps, batch_size, seq_width
    def train(self, batch_series_input, batch_series_expected_output, batch_series_output_valid):
        total_loss = 0
        for batch_num in range(len(batch_series_input)):
            feed_dict = {self.seq_input: batch_series_input[batch_num],
                         self.expected_output: batch_series_expected_output[batch_num],
                         self.expected_output_valid: batch_series_output_valid[batch_num]}

            self.trainer.run(session=self.session, feed_dict=feed_dict)
            total_loss += self.session.run(self.loss, feed_dict=feed_dict)

        return total_loss

    def predict(self, series_input):
        feed_dict = {self.seq_input: series_input}
        return self.session.run(self.outputs, feed_dict=feed_dict)

    def auto_train(self, num_iter=10, batch_length=80):
        total_loss = 0
        for iter_num in range(num_iter):
            batch = [NumberCounter.EMPTY_NUM]
            expected_out = [[0, 0]] * batch_length
            expected_out_valid = [0] * batch_length
            expected_out_valid[0] = 1
            num_one = 0
            num_two = 0

            for bt in range(1, batch_length):
                rnd = random.choice(NumberCounter.NUMS)
                if rnd == NumberCounter.EMPTY_NUM:
                    expected_out[bt - 1] = [num_one, num_two]
                    expected_out_valid[bt - 1] = 1
                    expected_out_valid[bt] = 1
                    num_one = 0
                    num_two = 0
                elif rnd == NumberCounter.ONE_NUM:
                    num_one += 1
                else:
                    num_two += 1
                batch.append(rnd)
            expected_out[-1] = [num_one, num_two]
            expected_out_valid[-1] = 1

            loss = self.train(batch, expected_out, expected_out_valid)
            total_loss += loss
            print('avg_loss:\t' + str(total_loss / float(iter_num)) + '\tlast_loss:\t' + str(loss))
        print('done...')
