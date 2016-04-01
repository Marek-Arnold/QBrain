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
        self.x = tf.placeholder(tf.float32, shape=[None, self.field_size, 3], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, self.field_size + 1], name='y')
        self.possible_moves = tf.placeholder(tf.float32, shape=[None, self.field_size + 1], name='possible_moves')

        self.constant_one = tf.constant(1.0)
        self.constant_zero = tf.constant(0.0)

        self.constant_error_multiplier = tf.constant(2.0)
        self.constant_error_power = tf.constant(2.0)

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

        W_fc_upper_bound = weight_variable([fully_connected_layers[-1], self.field_size + 1], "upper_bound_net")
        b_fc_upper_bound = bias_variable([self.field_size + 1], "upper_bound_net")

        W_fc_lower_bound = weight_variable([fully_connected_layers[-1], self.field_size + 1], "lower_bound_net")
        b_fc_lower_bound = bias_variable([self.field_size + 1], "lower_bound_net")

        lower_bound_net_variables = [W_fc_lower_bound, b_fc_lower_bound]
        self.savers['lower_bound_net'] = tf.train.Saver(lower_bound_net_variables)
        self.variables['lower_bound_net'] = lower_bound_net_variables

        upper_bound_net_variables = [W_fc_upper_bound, b_fc_upper_bound]
        self.savers['upper_bound_net'] = tf.train.Saver(upper_bound_net_variables)
        self.variables['upper_bound_net'] = upper_bound_net_variables

        y_mul = tf.minimum(tf.ceil(tf.abs(self.y)), self.constant_one)

        self.upper_bound_predicted_action_values = tf.nn.bias_add(tf.matmul(h_fc[-1], W_fc_upper_bound),
                                                                  b_fc_upper_bound)
        self.upper_bound_possible_predicted_action_values = tf.mul(self.upper_bound_predicted_action_values,
                                                                   self.possible_moves)
        self.upper_bound_errors_too_large = tf.maximum(self.constant_zero,
                                                       tf.mul(tf.sub(self.upper_bound_predicted_action_values, self.y),
                                                              y_mul))
        self.upper_bound_errors_too_small = tf.maximum(self.constant_zero,
                                                       tf.mul(tf.sub(self.y, self.upper_bound_predicted_action_values),
                                                              y_mul))
        self.upper_bound_errors = tf.add(self.upper_bound_errors_too_large,
                                         tf.pow(tf.mul(self.upper_bound_errors_too_small,
                                                       self.constant_error_multiplier),
                                                self.constant_error_power))
        self.upper_bound_error = tf.reduce_sum(self.upper_bound_errors)

        self.lower_bound_predicted_action_values = tf.nn.bias_add(tf.matmul(h_fc[-1], W_fc_lower_bound),
                                                                  b_fc_lower_bound)
        self.lower_bound_possible_predicted_action_values = tf.mul(self.lower_bound_predicted_action_values,
                                                                   self.possible_moves)
        self.lower_bound_errors_too_large = tf.maximum(self.constant_zero,
                                                       tf.mul(tf.sub(self.lower_bound_predicted_action_values, self.y),
                                                              y_mul))
        self.lower_bound_errors_too_small = tf.maximum(self.constant_zero,
                                                       tf.mul(tf.sub(self.y, self.lower_bound_predicted_action_values),
                                                              y_mul))
        self.lower_bound_errors = tf.add(self.lower_bound_errors_too_small,
                                         tf.pow(tf.mul(self.lower_bound_errors_too_large,
                                                       self.constant_error_multiplier),
                                                self.constant_error_power))
        self.lower_bound_error = tf.reduce_sum(self.lower_bound_errors)

        self.errors = tf.concat(0, [tf.reshape(self.lower_bound_error, [1]), tf.reshape(self.upper_bound_error, [1])])
        self.error = tf.add(self.lower_bound_error, self.upper_bound_error)
        self.predicted_lower_and_upper_bounds = tf.concat(1,
                                                          [self.lower_bound_possible_predicted_action_values,
                                                           self.upper_bound_possible_predicted_action_values])

        for var_name in self.variables:
            self.trainers[var_name] = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.error,
                                                                                          var_list=self.variables[
                                                                                              var_name])
        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.error)
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def predict(self, x_, possible_moves):
        bounds = self.sess.run(self.predicted_lower_and_upper_bounds,
                               feed_dict={self.x: x_, self.possible_moves: possible_moves})

        lower_bounds = bounds[0][:self.field_size + 1]
        upper_bounds = bounds[0][self.field_size + 1:]

        return lower_bounds, upper_bounds

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
            feed_dict = {self.x: x_, self.y: y_}

            trainer.run(session=self.sess, feed_dict=feed_dict)
            errors = self.sess.run(self.errors, feed_dict=feed_dict) / float(len(y_) / self.field_size)
            print('\t\terrors ' +
                  '\tlower: ' + str(errors[0]) +
                  '\tupper: ' + str(errors[1]) +
                  '\ttotal: ' + str(errors[0] + errors[1]))
            if errors[0] + errors[1] < max_error:
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
