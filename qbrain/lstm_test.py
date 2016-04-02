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
        self.num_steps = 80

        initializer = tf.random_uniform_initializer(-1, 1)

        self.seq_input = tf.placeholder(tf.float32, [self.num_steps, self.seq_width])
        self.expected_output = tf.placeholder(tf.float32, [self.num_steps, self.seq_width])
        self.expected_output_valid = tf.placeholder(tf.float32, [self.num_steps])

        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
        self.initial_state = self.cell.zero_state(1, tf.float32)

        # ========= This is the most important part ==========
        # output will be of length 4 and 6
        # the state is the final state at termination (stopped at step 4 and 6)

        inp = [tf.reshape(i, (1, self.seq_width)) for i in tf.split(0, self.num_steps, self.seq_input)]
        self.outputs, self.state = tf.nn.rnn(self.cell, inp, initial_state=self.initial_state)
        output = tf.reshape(tf.concat(1, self.outputs), [-1, self.lstm_size])
        softmax_w = tf.get_variable("softmax_w", [self.lstm_size, 2])
        softmax_b = tf.get_variable("softmax_b", [2])
        logits = tf.nn.bias_add(tf.matmul(output, softmax_w), softmax_b)

        # self.loss = tf.reduce_sum(tf.mul(tf.sub(logits, self.expected_output), tf.reshape(tf.concat(0, [self.expected_output_valid, self.expected_output_valid]), [-1, 2])))

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.expected_output)

        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
        # usual crap
        iop = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(iop)

    # batch_series_input: n_steps, batch_size, seq_width
    def train(self, batch_series_input, batch_series_expected_output, batch_series_output_valid):
        # total_loss = 0
        # for batch_num in range(len(batch_series_input)):
        # feed_dict = {self.seq_input: [batch_series_input[batch_num]],
        # self.expected_output: [batch_series_expected_output[batch_num]],
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

    def auto_train(self, num_iter=10, batch_length=80, echo=False):
        total_loss = 0
        for iter_num in range(num_iter):
            batch = [NumberCounter.EMPTY_NUM]
            expected_out = [NumberCounter.EMPTY_NUM] * batch_length
            expected_out_valid = [0] * batch_length
            expected_out_valid[0] = 1
            num_one = 0
            num_two = 0

            for bt in range(1, batch_length):
                rnd = random.choice(NumberCounter.NUMS)
                if rnd == NumberCounter.ONE_NUM:
                    num_one += 1
                elif rnd == NumberCounter.TWO_NUM:
                    num_two += 1

                batch.append(rnd)
            expected_out[-1] = batch[-1]
            expected_out_valid[-1] = 1

            loss = self.train(batch, expected_out, expected_out_valid)
            total_loss += loss
            print('avg_loss:\t' + str(total_loss / float(iter_num)) + '\tlast_loss:\t' + str(loss))

            if echo:
                pred = self.predict(batch)
                for i in range(len(batch)):
                    print(str(batch[i]) + '\t' + str(expected_out[i]) + '\t' + str(pred[i][0][0]) + ', ' + str(
                        pred[i][0][1]))
                input('Press enter to continue..')
        print('done...')
