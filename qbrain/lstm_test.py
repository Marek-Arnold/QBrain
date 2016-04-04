__author__ = 'Marek'
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell
import random
import math


class NumberCounter:
    EMPTY_NUM = [0, 0]
    ONE_NUM = [1, 0]
    TWO_NUM = [0, 1]
    STOP_WORD = [1, 1]

    IN_WORD = [1, 0, 0]
    VALID_WORD_END = [0, 1, 0]
    INVALID_WORD_END = [0, 0, 1]

    def __init__(self, seq_width=2):
        self.lstm_size = 16
        self.lstm_layers = 4
        self.seq_width = seq_width
        self.num_steps = 1
        self.num_out = 3

        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=2.0)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * self.lstm_layers)

        self.batch_size = tf.placeholder(tf.int32, [1])
        batch_size = self.batch_size[0]
        self.seq_input = tf.placeholder(tf.float32, [None, self.num_steps, self.seq_width])
        self.state_input = tf.placeholder(tf.float32, [None, stacked_lstm.state_size])
        self.expected_output = tf.placeholder(tf.float32, [None, self.num_steps, self.num_out])

        self.initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

        # ========= This is the most important part ==========
        # output will be of length 4 and 6
        # the state is the final state at termination (stopped at step 4 and 6)

        inp = [tf.squeeze(i, [1]) for i in tf.split(1, self.num_steps, self.seq_input)]
        outputs, self.final_state = tf.nn.rnn(stacked_lstm, inp, initial_state=self.state_input)
        output = tf.reshape(tf.concat(1, outputs), [-1, self.lstm_size])
        softmax_w = tf.get_variable("softmax_w", [self.lstm_size, self.num_out])
        softmax_b = tf.get_variable("softmax_b", [self.num_out])
        logits = tf.nn.bias_add(tf.matmul(output, softmax_w), softmax_b)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits, tf.reshape(self.expected_output, [-1, self.num_out]))
        self.total_loss = tf.reduce_sum(self.loss)

        self.predictions = tf.reshape(tf.nn.softmax(logits), [-1, self.num_steps, self.num_out])
        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
        # usual crap
        iop = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(iop)

    # batch_series_input: n_steps, batch_size, seq_width
    def train(self, batch_series_input, batch_series_expected_output):
        total_loss = 0
        state = self.initial_state.eval(session=self.session, feed_dict={self.batch_size: [len(batch_series_input)]})
        for i in range(int(len(batch_series_input[0]) / self.num_steps)):
            mini_batch = []
            mini_batch_expected = []
            for batch_num in range(len(batch_series_input)):
                start = i * self.num_steps
                end = (i + 1) * self.num_steps
                mini_batch.append(batch_series_input[batch_num][start:end])
                mini_batch_expected.append(batch_series_expected_output[batch_num][start:end])

            feed_dict = {self.batch_size: [len(batch_series_input)],
                         self.seq_input: mini_batch,
                         self.expected_output: mini_batch_expected,
                         self.state_input: state}

            self.trainer.run(session=self.session, feed_dict=feed_dict)
            loss, state = self.session.run([self.total_loss, self.final_state], feed_dict=feed_dict)
            total_loss += loss

        return total_loss

    def predict(self, series_input):
        state = self.initial_state.eval(session=self.session, feed_dict={self.batch_size: [1]})
        res = []
        for i in range(int(math.ceil(len(series_input) / float(self.num_steps)))):
            start = i * self.num_steps
            end = (i + 1) * self.num_steps
            feed_dict = {self.batch_size: [1],
                         self.seq_input: [series_input[start:end]],
                         self.state_input: state}
            rp, state = self.session.run([self.predictions, self.final_state], feed_dict=feed_dict)
            res.extend(rp[0])

        return res

    def gen_batch(self, batch_length, max_word_length):
        batch = [None] * batch_length
        expected_out = [NumberCounter.IN_WORD] * batch_length

        ind = 0
        while ind < batch_length:
            if batch_length - ind <= 5:
                for i in range(batch_length - ind):
                    batch[ind] = random.choice([NumberCounter.ONE_NUM, NumberCounter.TWO_NUM])
                    ind += 1
            else:
                correct_word = random.random() > 0.5

                num_chars = min(int(max_word_length / 2), random.randint(1, int((batch_length - ind) / 2)))
                if correct_word:
                    for i in range(num_chars):
                        batch[ind] = NumberCounter.ONE_NUM
                        ind += 1
                    for i in range(num_chars):
                        batch[ind] = NumberCounter.TWO_NUM
                        ind += 1
                    if ind < batch_length:
                        expected_out[ind] = NumberCounter.VALID_WORD_END
                else:
                    if random.random() > 0.5:
                        rnd = 1
                    else:
                        rnd = -1

                    for i in range(num_chars + rnd):
                        batch[ind] = NumberCounter.ONE_NUM
                        ind += 1
                    for i in range(num_chars - rnd):
                        batch[ind] = NumberCounter.TWO_NUM
                        ind += 1

                    if ind < batch_length:
                        expected_out[ind] = NumberCounter.INVALID_WORD_END

                if ind < batch_length:
                    batch[ind] = NumberCounter.STOP_WORD
                    ind += 1

        return batch, expected_out

    def auto_train(self, num_iter=10, max_word_length=50, batch_length=400, num_batches=10, echo=False, loss_print_iter=100):
        total_loss = 0
        for iter_num in range(num_iter):
            batches = [None] * num_batches
            expected_outs = [None] * num_batches

            num_valid = 0
            num_inword = 0
            num_invalid = 0
            num_correct_valid = 0
            num_correct_invalid = 0
            num_correct_inword = 0

            for batch_num in range(num_batches):
                batch, expected_out = self.gen_batch(batch_length=batch_length, max_word_length=max_word_length)

                batches[batch_num] = batch
                expected_outs[batch_num] = expected_out
                prediction = self.predict(batch)
                print(prediction)

                for i in range(len(batch)):
                    if expected_out[i] == NumberCounter.VALID_WORD_END:
                        num_valid += 1
                        if list(prediction[i]) == NumberCounter.VALID_WORD_END:
                            num_correct_valid += 1
                    elif expected_out[i] == NumberCounter.INVALID_WORD_END:
                        num_invalid += 1
                        if list(prediction[i]) == NumberCounter.INVALID_WORD_END:
                            num_correct_invalid += 1
                    else:
                        num_inword += 1
                        if list(prediction[i]) == NumberCounter.IN_WORD:
                            num_correct_inword += 1

            loss = self.train(batches, expected_outs)
            total_loss += loss
            print('avg_loss:\t' + str(total_loss / float(iter_num + 1)) + '\tlast_loss:\t' + str(loss) +
                  '\tvalid:\t' + str(num_correct_valid) + '/' + str(num_valid) +
                  '\tinvalid:\t' + str(num_correct_invalid) + '/' + str(num_invalid) +
                  '\tinword:\t' + str(num_correct_inword) + '/' + str(num_inword))

        print('avg_loss:\t' + str(total_loss / float(num_iter))) #  + '\tlast_loss:\t' + str(loss))
        print('done...')

    def eval(self, max_word_length=50, batch_length=400):
        batch, expected_out = self.gen_batch(batch_length=batch_length, max_word_length=max_word_length)
        pred = self.predict(batch)

        for i in range(len(batch)):
            print(str(batch[i]) + '\t' + str(expected_out[i]) + '\t' + str(pred[i]))
