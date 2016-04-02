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
        self.num_steps = 8

        initializer = tf.random_uniform_initializer(-1, 1)

        self.seq_input = tf.placeholder(tf.float32, [self.num_steps, self.seq_width])
        self.expected_output = tf.placeholder(tf.float32, [self.num_steps, self.seq_width])
        self.expected_output_valid = tf.placeholder(tf.float32, [self.num_steps])

        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.seq_width, forget_bias=2.0)
        self.initial_state = self.cell.zero_state(1, tf.float32)

        # ========= This is the most important part ==========
        # output will be of length 4 and 6
        # the state is the final state at termination (stopped at step 4 and 6)

        self.state = self.initial_state
        inp = [tf.reshape(i, (1, self.seq_width)) for i in tf.split(0, self.num_steps, self.seq_input)]
        self.outputs, self.state = tf.nn.rnn(self.cell, inp, initial_state=self.state)

        self.loss = tf.reduce_sum(tf.mul(tf.reduce_sum(tf.pow(tf.sub(self.outputs[0], self.expected_output), 2)), self.expected_output_valid))

        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
        # usual crap
        iop = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(iop)

    # batch_series_input: n_steps, batch_size, seq_width
    def train(self, batch_series_input, batch_series_expected_output, batch_series_output_valid):
        # total_loss = 0
        # for batch_num in range(len(batch_series_input)):
        #     feed_dict = {self.seq_input: [batch_series_input[batch_num]],
        #                  self.expected_output: [batch_series_expected_output[batch_num]],
        #                  self.expected_output_valid: [batch_series_output_valid[batch_num]]}
        #
        #     self.trainer.run(session=self.session, feed_dict=feed_dict)
        #     total_loss += self.session.run(self.loss, feed_dict=feed_dict)
        #
        # return total_loss
        feed_dict = {self.seq_input: batch_series_input,
                     self.expected_output: batch_series_expected_output,
                     self.expected_output_valid: batch_series_output_valid}

        self.trainer.run(session=self.session, feed_dict=feed_dict)
        return self.session.run(self.loss, feed_dict=feed_dict)

    def predict(self, series_input):
        feed_dict = {self.seq_input: series_input}
        return self.session.run(self.outputs, feed_dict=feed_dict)

    def auto_train(self, num_iter=10, batch_length=8):
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
                    num_one = 0
                    num_two = 0
                elif rnd == NumberCounter.ONE_NUM:
                    num_one += 1
                else:
                    num_two += 1
                expected_out[bt] = [num_one, num_two]
                expected_out_valid[bt] = 1
                batch.append(rnd)

            loss = self.train(batch, expected_out, expected_out_valid)
            total_loss += loss
            print('avg_loss:\t' + str(total_loss / float(iter_num)) + '\tlast_loss:\t' + str(loss))

            pred = self.predict(batch)
            for i in range(len(batch)):
                print(str(batch[i]) + '\t' + str(expected_out[i]) + '\t' + str(pred[i][0][0]) + ', ' + str(pred[i][0][1]))
            input('Press enter to continue..')
        print('done...')
