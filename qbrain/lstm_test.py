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
    VALID_WORD = [1, 0]
    INVALID_WORD = [0, 1]
    STOP_WORD = [1, 1]

    def __init__(self, seq_width=2):
        self.lstm_size = 64
        self.lstm_layers = 2
        self.seq_width = seq_width
        self.num_steps = 4

        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=1.0)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * self.lstm_layers)

        self.seq_input = tf.placeholder(tf.float32, [self.num_steps, self.seq_width])
        self.state_input = tf.placeholder(tf.float32, [1, stacked_lstm.state_size])
        self.expected_output = tf.placeholder(tf.float32, [self.num_steps, self.seq_width])

        self.initial_state = stacked_lstm.zero_state(1, tf.float32)

        # ========= This is the most important part ==========
        # output will be of length 4 and 6
        # the state is the final state at termination (stopped at step 4 and 6)

        inp = [tf.reshape(i, (1, self.seq_width)) for i in tf.split(0, self.num_steps, self.seq_input)]
        outputs, self.final_state = tf.nn.rnn(stacked_lstm, inp, initial_state=self.state_input)
        output = tf.reshape(tf.concat(1, outputs), [-1, self.lstm_size])
        softmax_w = tf.get_variable("softmax_w", [self.lstm_size, 2])
        softmax_b = tf.get_variable("softmax_b", [2])
        logits = tf.nn.bias_add(tf.matmul(output, softmax_w), softmax_b)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits, self.expected_output)
        self.total_loss = tf.reduce_sum(self.loss)

        self.predictions = tf.nn.softmax(logits)
        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
        # usual crap
        iop = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(iop)

    # batch_series_input: n_steps, batch_size, seq_width
    def train(self, batch_series_input, batch_series_expected_output):
        total_loss = 0
        state = self.initial_state.eval(session=self.session)
        for i in range(int(len(batch_series_input) / self.num_steps)):
            start = i * self.num_steps
            end = (i + 1) * self.num_steps
            feed_dict = {self.seq_input: batch_series_input[start:end],
                         self.expected_output: batch_series_expected_output[start:end],
                         self.state_input: state}

            self.trainer.run(session=self.session, feed_dict=feed_dict)
            loss, state = self.session.run([self.total_loss, self.final_state], feed_dict=feed_dict)
            total_loss += loss

        return total_loss

    def predict(self, series_input):
        state = self.initial_state.eval(session=self.session)
        res = []
        for i in range(int(math.ceil(len(series_input) / float(self.num_steps)))):
            start = i * self.num_steps
            end = (i + 1) * self.num_steps
            feed_dict = {self.seq_input: series_input[start:end],
                         self.state_input: state}
            rp, state = self.session.run([self.predictions, self.final_state], feed_dict=feed_dict)
            res.extend(rp)

        return res

    def auto_train(self, num_iter=10, batch_length=400, echo=False, loss_print_iter=100):
        total_loss = 0
        for iter_num in range(num_iter):
            batch = [NumberCounter.EMPTY_NUM]
            expected_out = [NumberCounter.EMPTY_NUM] * batch_length
            num_one = 0
            num_two = 0

            for bt in range(1, batch_length):
                rnd = random.choice([NumberCounter.ONE_NUM, NumberCounter.TWO_NUM])
                if rnd == NumberCounter.ONE_NUM:
                    num_one += 1
                elif rnd == NumberCounter.TWO_NUM:
                    num_two += 1

                batch.append(rnd)
                expected_out[bt] = batch[bt-1]

            if echo:
                pred = self.predict(batch)
                for i in range(len(batch)):
                    print(str(batch[i]) + '\t' + str(expected_out[i]) + '\t' + str(pred[i]))
                input('Press enter to continue..')

            loss = self.train(batch, expected_out)
            total_loss += loss
            print('avg_loss:\t' + str(total_loss / float(iter_num)) + '\tlast_loss:\t' + str(loss))

        print('avg_loss:\t' + str(total_loss / float(num_iter)))
        print('done...')

    def auto_train2(self, num_iter=10, max_word_length=50, batch_length=400, echo=False, loss_print_iter=100):
        total_loss = 0
        for iter_num in range(num_iter):
            batch = [None] * batch_length
            expected_out = [NumberCounter.INVALID_WORD] * batch_length

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
                            expected_out[ind] = NumberCounter.VALID_WORD
                    else:
                        if random.random() > 0.5:
                            rnd = 1.0
                        else:
                            rnd = -1.0

                        for i in range(num_chars + rnd):
                            batch[ind] = NumberCounter.ONE_NUM
                            ind += 1
                        for i in range(num_chars - rnd):
                            batch[ind] = NumberCounter.TWO_NUM
                            ind += 1

                    if ind < batch_length:
                        batch[ind] = NumberCounter.STOP_WORD
                        ind += 1

            pred = self.predict(batch)
            num_valid = 0
            num_invalid = 0
            num_correct_valid = 0
            num_correct_invalid = 0
            for i in range(len(batch)):
                if expected_out[i] == NumberCounter.VALID_WORD:
                    num_valid += 1
                    if pred[i][0] > pred[i][1]:
                        num_correct_valid += 1
                else:
                    num_invalid += 1
                    if pred[i][1] > pred[i][0]:
                        num_correct_invalid += 1

            loss = self.train(batch, expected_out)
            total_loss += loss
            print('avg_loss:\t' + str(total_loss / float(iter_num)) + '\tlast_loss:\t' + str(loss) + '\tvalid:\t' + str(num_correct_valid) + '/' + str(num_valid) + '\tinvalid:\t' + str(num_correct_invalid) + '/' + str(num_invalid))

            if echo:
                for i in range(len(batch)):
                    print(str(batch[i]) + '\t' + str(expected_out[i]) + '\t' + str(pred[i]))
                input('Press enter to continue..')
        print('avg_loss:\t' + str(total_loss / float(num_iter))) #  + '\tlast_loss:\t' + str(loss))
        print('done...')
