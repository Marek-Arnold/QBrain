import tensorflow as tf
import os


class QBrainGoNet:
    """
    Deep artificial neuronal network based on tensorflow.
    """

    def __init__(self,
                 board_size,
                 convolution_layers,
                 fully_connected_layers):
        """
        Parameters
        ----------
        :param: board_size: int
        :param: convolution_layers: list of tuple of int and int
        Each tuple in this list holds the information for a convolution layer.
        0: int
            input size
        1: int
            output size

        :param: fully_connected_layers: list of int


        Returns
        -------
        :return: QBrainGoNet
            A freshly initialized tensorflow network.
        """

        self.board_size = board_size
        self.field_size = board_size * board_size

        self.savers = {}
        self.variables = {}
        self.trainers = {}

        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape=[None, self.field_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.field_size + 1])
        self.possible_moves = tf.placeholder(tf.float32, shape=[None, self.field_size + 1])

        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial, name="weights_" + name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name="bias_" + name)

        reshaped_x = tf.reshape(self.x, [-1, self.board_size, self.board_size, 1])

        W_conv = [None] * len(convolution_layers)
        b_conv = [None] * len(convolution_layers)
        h_conv = [None] * len(convolution_layers)

        for conv_layer_num in range(len(convolution_layers)):
            conv_layer = convolution_layers[conv_layer_num]

            if conv_layer_num == 0:
                input_channels = 1
                input_data = reshaped_x
            else:
                input_channels = convolution_layers[conv_layer_num - 1][1]
                input_data = h_conv[conv_layer_num - 1]
            output_channels = conv_layer[1]

            patch_size = conv_layer[0]

            W_conv[conv_layer_num] = weight_variable([patch_size, patch_size, input_channels, output_channels],
                                                     'conv_layer_' + str(conv_layer_num))
            b_conv[conv_layer_num] = bias_variable([output_channels], 'conv_layer_' + str(conv_layer_num))
            h_conv[conv_layer_num] = tf.nn.relu(tf.nn.bias_add(
                tf.nn.conv2d(input_data, W_conv[conv_layer_num], strides=[1, 1, 1, 1], padding='SAME'),
                b_conv[conv_layer_num]))

            conv_layer_variables = [W_conv[conv_layer_num], b_conv[conv_layer_num]]
            self.savers['conv_layer_' + str(conv_layer_num)] = tf.train.Saver(conv_layer_variables)
            self.variables['conv_layer_' + str(conv_layer_num)] = conv_layer_variables

        num_features = convolution_layers[-1][1] * self.field_size
        h_conv_reshaped = tf.reshape(h_conv[-1], [-1, num_features])

        W_fc = [None] * len(fully_connected_layers)
        b_fc = [None] * len(fully_connected_layers)
        h_fc = [None] * len(fully_connected_layers)

        for fc_layer_num in range(len(fully_connected_layers)):
            if fc_layer_num == 0:
                fc_input = h_conv_reshaped
                fc_input_size = num_features
            else:
                fc_input = h_fc[fc_layer_num - 1]
                fc_input_size = fully_connected_layers[fc_layer_num - 1]
            fc_output_size = fully_connected_layers[fc_layer_num]

            W_fc[fc_layer_num] = weight_variable([fc_input_size, fc_output_size], "fc_" + str(fc_layer_num))
            b_fc[fc_layer_num] = bias_variable([fc_output_size], "fc_" + str(fc_layer_num))
            h_fc[fc_layer_num] = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc_input, W_fc[fc_layer_num]), b_fc[fc_layer_num]))

            fully_connected_net_variables = [W_fc[fc_layer_num], b_fc[fc_layer_num]]
            self.savers['fully_connected_layer_' + str(fc_layer_num)] = tf.train.Saver(fully_connected_net_variables)
            self.variables['fully_connected_layer_' + str(fc_layer_num)] = fully_connected_net_variables

        W_fc_last = weight_variable([fully_connected_layers[-1], self.field_size], "action_net")
        b_fc_last = bias_variable([self.field_size], "action_net")

        action_net_variables = [W_fc_last, b_fc_last]
        self.savers['action_net'] = tf.train.Saver(action_net_variables)
        self.variables['action_net'] = action_net_variables

        self.predicted_action_values = tf.mul(tf.reshape(tf.nn.bias_add(tf.matmul(h_fc[-1], W_fc_last), b_fc_last),
                                                         [-1, self.board_size, self.board_size, 1]),
                                              self.possible_moves)

        self.errors = tf.reduce_sum(tf.mul(tf.abs((self.y_ - self.predicted_action_values), self.y_)))

        for var_name in self.variables:
            self.trainers[var_name] = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.errors,
                                                                                          var_list=self.variables[
                                                                                              var_name])
        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.errors)
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def predict(self, x_):
        """
        Predict the q-values for the given input.
        Parameters
        ----------
        :param x_: list of float
            A series of observations.
        :return: list of float
            The predicted q-values for each action.
        """
        return self.sess.run(self.predicted_action_values, feed_dict={self.x: x_})

    def train(self, x_, y_, num_iterations, max_error, train_layer_name):
        """
        Parameters
        ----------
        :param x_: list of list of float
            List of series of observations.
        :param y_: list of list of float
            List of one hot rewards for the observation series.
        :param num_iterations:
            The number of iterations to train.

        Returns
        -------
        :return: None
        """

        if train_layer_name is None:
            trainer = self.trainer
        else:
            trainer = self.trainers[train_layer_name]

        for i in range(num_iterations):
            feed_dict = {self.x: x_, self.y_: y_}

            trainer.run(session=self.sess, feed_dict=feed_dict)
            error = self.sess.run(self.errors, feed_dict=feed_dict) / float(len(y_) / self.field_size)
            print('\t\tloss: ' + str(error))
            if error < max_error:
                break

    def save_multi_file(self, model_base_name, extension):
        for saver_name in self.savers:
            file_name = model_base_name + '_' + saver_name + extension
            print('saving:\t' + str(self.savers[saver_name].save(self.sess, file_name)))
        print('saved net')

    def load_multi_file(self, model_base_name, extension):
        for saver_name in self.savers:
            file_name = model_base_name + '_' + saver_name + extension
            if os.path.exists(file_name):
                self.savers[saver_name].restore(self.sess, file_name)
                print('restored:\t' + file_name)
            else:
                print('not_found:\t' + file_name)
        print('loaded net')
