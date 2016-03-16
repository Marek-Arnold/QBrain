import tensorflow as tf


class QBrainNet:
    """
    Deep artificial neuronal network based on tensorflow.
    """
    def __init__(self,
                 single_input_size,
                 temporal_window_size,
                 num_actions,
                 num_neurons_in_convolution_layers,
                 num_neurons_in_fully_connected_layers):
        """
        Parameters
        ----------
        :param single_input_size: int
            Size of a single observation including one hot action.
        :param temporal_window_size: int
            Number of observations seen at once.
        :param num_actions: int
            Number of actions that can be taken.
        :param num_neurons_in_convolution_layers: list of int
            Number of features in the convolution layers that should be learned for the representation of a single input.
        :param num_neurons_in_fully_connected_layers: list of int
            Number of neurons in the fully connected layers.

        Returns
        -------
        :return: QBrainNet
            A freshly initialized tensorflow network.
        """
        self.num_inputs_total = single_input_size * temporal_window_size
        self.num_actions = num_actions

        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape=[None, self.num_inputs_total])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_actions])

        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name="weights_" + name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name="bias_" + name)

        input_conv = [None] * len(num_neurons_in_convolution_layers)
        W_conv = [None] * len(num_neurons_in_convolution_layers)
        b_conv = [None] * len(num_neurons_in_convolution_layers)
        h_conv = [None] * len(num_neurons_in_convolution_layers)
        h_conv_reshaped = [None] * len(num_neurons_in_convolution_layers)

        for conv_layer_num in range(len(num_neurons_in_convolution_layers)):
            input_size = single_input_size
            if conv_layer_num == 0:
                input_conv[conv_layer_num] = tf.reshape(self.x, [-1, 1, temporal_window_size, single_input_size])
            else:
                input_size = num_neurons_in_convolution_layers[conv_layer_num - 1]
                input_conv[conv_layer_num] = tf.reshape(h_conv_reshaped[conv_layer_num - 1], [-1, 1, temporal_window_size, input_size])

            W_conv[conv_layer_num] = weight_variable([1, 1, input_size, num_neurons_in_convolution_layers[conv_layer_num]], "convolution_" + str(conv_layer_num))
            b_conv[conv_layer_num] = bias_variable([num_neurons_in_convolution_layers[conv_layer_num]], "convolution_" + str(conv_layer_num))
            h_conv[conv_layer_num] = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_conv[conv_layer_num], W_conv[conv_layer_num], strides=[1, 1, 1, 1], padding='SAME'), b_conv[conv_layer_num]))

            h_conv_reshaped[conv_layer_num] = tf.reshape(h_conv[conv_layer_num], [-1, temporal_window_size * num_neurons_in_convolution_layers[conv_layer_num]])


        W_fc = [None] * len(num_neurons_in_fully_connected_layers)
        b_fc = [None] * len(num_neurons_in_fully_connected_layers)
        h_fc = [None] * len(num_neurons_in_fully_connected_layers)

        W_fc[0] = weight_variable([num_neurons_in_convolution_layers[-1] * temporal_window_size, num_neurons_in_fully_connected_layers[0]], "fc_0")
        b_fc[0] = bias_variable([num_neurons_in_fully_connected_layers[0]], "fc_0")
        h_fc[0] = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_conv_reshaped[-1], W_fc[0]), b_fc[0]))

        for layerNum in range(1, len(num_neurons_in_fully_connected_layers)):
            W_fc[layerNum] = weight_variable([num_neurons_in_fully_connected_layers[layerNum - 1], num_neurons_in_fully_connected_layers[layerNum]], "fc_" + str(layerNum))
            b_fc[layerNum] = bias_variable([num_neurons_in_fully_connected_layers[layerNum]], "fc_" + str(layerNum))
            h_fc[layerNum] = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc[layerNum - 1], W_fc[layerNum]), b_fc[layerNum]))

        W_fc_last = weight_variable([num_neurons_in_fully_connected_layers[-1], num_actions], "fc_last")
        b_fc_last = bias_variable([num_actions], "fc_last")

        self.predicted_action_values = tf.nn.bias_add(tf.matmul(h_fc[-1], W_fc_last), b_fc_last)

        self.errors = tf.reduce_sum(tf.abs((self.y_ - self.predicted_action_values) * self.y_))

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.errors)
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

    def train(self, x_, y_, num_iterations):
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
        for i in range(num_iterations):
            feed_dict = {self.x: x_, self.y_: y_}
            self.train_step.run(session=self.sess, feed_dict=feed_dict)
            print('\t\tloss: ' + str(self.sess.run(self.errors, feed_dict=feed_dict) / float(len(y_) / self.num_actions)))

    def save(self, model_name):
        """
        Save the model.

        Parameters
        ----------
        :param model_name: str
            The path to the model file.

        Returns
        -------
        :return: None
        """
        print(self.saver.save(self.sess, model_name))

    def load(self, model_name):
        """
        Load a previously saved model.

        Parameters
        ----------
        :param model_name: str
            The path to the model file.

        Returns
        -------
        :return: None
        """
        self.saver.restore(self.sess, model_name)
