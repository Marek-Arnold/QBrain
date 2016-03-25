import tensorflow as tf
import os


class QBrainNet:
    """
    Deep artificial neuronal network based on tensorflow.
    """
    def __init__(self,
                 single_input_size,
                 temporal_window_size,
                 num_actions,
                 sensor_descriptions,
                 num_neurons_in_convolution_layers,
                 num_neurons_in_convolution_layers_for_time,
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
        :param sensor_descriptions: list of tuple of int and int and list of int
            List of tuples that describe the sensors [(num_sensors, num_inputs_per_sensor, network_per_sensor, network_for_sensors, sensor_name)]
            num_sensors: Number of sensors of this kind.
            num_inputs_per_sensor: Number of features one sensor observes.
            network_per_sensor: List of int to describe the network to use for the processing of this single sensor.
            network_for_sensors: List of int to describe the network to use for all sensors of this kind in one time step.
            sensor_name: Unique name of this sensor.
        :param num_neurons_in_convolution_layers: list of int
            Number of features in the convolution layers that should be learned for the representation of a single input.
        :param num_neurons_in_convolution_layers_for_time: list of int
            Number of features in the convolution layers that should be learned for the representation of two single inputs.
        :param num_neurons_in_fully_connected_layers: list of int
            Number of neurons in the fully connected layers.

        Returns
        -------
        :return: QBrainNet
            A freshly initialized tensorflow network.
        """
        self.num_inputs_total = single_input_size * temporal_window_size
        self.num_actions = num_actions
        self.sensor_descriptions = sensor_descriptions
        self.savers = {}
        self.variables = {}

        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape=[None, self.num_inputs_total])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_actions])

        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.03)
            return tf.Variable(initial, name="weights_" + name)

        def bias_variable(shape, name):
            initial = tf.constant(0.001, shape=shape)
            return tf.Variable(initial, name="bias_" + name)

        sensor_offsets = [0] * (len(sensor_descriptions) + 1)
        adapted_sensor_data = [None] * len(sensor_descriptions)
        adapted_sensor_data_group_sizes = [0] * len(sensor_descriptions)

        for sensor_description_num in range(len(sensor_descriptions)):
            sensor_description = sensor_descriptions[sensor_description_num]
            ix = sensor_description_num

            num_sensors = sensor_description[0]
            single_sensor_size = sensor_description[1]
            single_sensor_net = sensor_description[2]
            sensor_group_net = sensor_description[3]
            sensor_name = sensor_description[4]

            sensor_group_size = num_sensors * single_sensor_size

            sensor_offsets[ix + 1] = sensor_group_size + sensor_offsets[ix]

            sensor_group = None

            for temporal_window_num in range(temporal_window_size):
                sliced_sensor = tf.slice(self.x, [0, sensor_offsets[ix] + single_input_size * temporal_window_num], [-1, sensor_group_size])
                if sensor_group is None:
                    sensor_group = sliced_sensor
                else:
                    sensor_group = tf.concat(1, [sensor_group, sliced_sensor])

            if len(single_sensor_net) > 0:
                W_single_sensor = [None] * len(single_sensor_net)
                b_single_sensor = [None] * len(single_sensor_net)
                h_single_sensor = [None] * len(single_sensor_net)

                reshaped_group = tf.reshape(sensor_group, [-1, 1, 1, single_sensor_size])

                for layer_num in range(0, len(single_sensor_net)):
                    if layer_num == 0:
                        input_size = single_sensor_size
                        sensor_input = reshaped_group
                    else:
                        input_size = single_sensor_net[layer_num - 1]
                        sensor_input = h_single_sensor[layer_num - 1]
                    output_size = single_sensor_net[layer_num]

                    W_single_sensor[layer_num] = weight_variable([1, 1, input_size, output_size],
                                                                 "single_sensor_" + sensor_name + "_layer_" + str(layer_num))
                    b_single_sensor[layer_num] = bias_variable([single_sensor_net[layer_num]],
                                                               "single_sensor_" + sensor_name + "_layer_" + str(layer_num))
                    h_single_sensor[layer_num] = tf.nn.relu(tf.nn.bias_add(
                        tf.nn.conv2d(sensor_input, W_single_sensor[layer_num], strides=[1, 1, 1, 1], padding='SAME'),
                        b_single_sensor[layer_num]))

                single_sensor_variables = []
                single_sensor_variables.extend(W_single_sensor)
                single_sensor_variables.extend(b_single_sensor)
                self.savers[sensor_name + '_single_sensor'] = tf.train.Saver(single_sensor_variables)
                self.variables[sensor_name + '_single_sensor'] = single_sensor_variables

                sensor_group_size = num_sensors * single_sensor_net[-1]
                sensor_group = tf.reshape(h_single_sensor[-1], [-1, sensor_group_size * temporal_window_size])

            if len(sensor_group_net) > 0:
                W_sensor_group = [None] * len(sensor_group_net)
                b_sensor_group = [None] * len(sensor_group_net)
                h_sensor_group = [None] * len(sensor_group_net)

                reshaped_group = tf.reshape(sensor_group, [-1, 1, 1, sensor_group_size])

                for layer_num in range(0, len(sensor_group_net)):
                    if layer_num == 0:
                        input_size = sensor_group_size
                        sensor_input = reshaped_group
                    else:
                        input_size = sensor_group_net[layer_num - 1]
                        sensor_input = h_sensor_group[layer_num - 1]
                    output_size = sensor_group_net[layer_num]

                    W_sensor_group[layer_num] = weight_variable([1, 1, input_size, output_size],
                                                                "sensor_group_" + sensor_name + "_layer_" + str(layer_num))
                    b_sensor_group[layer_num] = bias_variable([output_size],
                                                              "sensor_group_" + sensor_name + "_layer_" + str(layer_num))
                    h_sensor_group[layer_num] = tf.nn.relu(tf.nn.bias_add(
                        tf.nn.conv2d(sensor_input, W_sensor_group[layer_num], strides=[1, 1, 1, 1], padding='SAME'),
                        b_sensor_group[layer_num]))

                sensor_group_variables = []
                sensor_group_variables.extend(W_sensor_group)
                sensor_group_variables.extend(b_sensor_group)
                self.savers[sensor_name + '_sensor_group'] = tf.train.Saver(sensor_group_variables)
                self.variables[sensor_name + '_sensor_group'] = sensor_group_variables

                sensor_group_size = sensor_group_net[-1]
                sensor_group = tf.reshape(h_sensor_group[-1], [-1, sensor_group_size * temporal_window_size])

            adapted_sensor_data[ix] = sensor_group
            adapted_sensor_data_group_sizes[ix] = sensor_group_size

        adaptedx = None
        for temporal_window_num in range(temporal_window_size):
            for sensor_num in range(0, len(sensor_descriptions)):
                sensor_group_size = adapted_sensor_data_group_sizes[sensor_num]
                sliced_adapted_data = tf.slice(adapted_sensor_data[sensor_num],
                                               [0, sensor_group_size * temporal_window_num], [-1, sensor_group_size])
                if adaptedx is None:
                    adaptedx = sliced_adapted_data
                else:
                    adaptedx = tf.concat(1, [adaptedx, sliced_adapted_data])

        adapted_input_size = 0
        for sensor_num in range(0, len(sensor_descriptions)):
            adapted_input_size += adapted_sensor_data_group_sizes[sensor_num]

        W_conv = [None] * len(num_neurons_in_convolution_layers)
        b_conv = [None] * len(num_neurons_in_convolution_layers)
        h_conv = [None] * len(num_neurons_in_convolution_layers)

        for conv_layer_num in range(len(num_neurons_in_convolution_layers)):
            if conv_layer_num == 0:
                input_size = adapted_input_size
                input_conv = tf.reshape(adaptedx, [-1, 1, temporal_window_size, input_size])
            else:
                input_size = num_neurons_in_convolution_layers[conv_layer_num - 1]
                input_conv = tf.reshape(h_conv[conv_layer_num - 1],
                                                        [-1, 1, temporal_window_size, input_size])

            W_conv[conv_layer_num] = weight_variable([1, 1, input_size, num_neurons_in_convolution_layers[conv_layer_num]],
                                                     "convolution_" + str(conv_layer_num))
            b_conv[conv_layer_num] = bias_variable([num_neurons_in_convolution_layers[conv_layer_num]],
                                                   "convolution_" + str(conv_layer_num))
            h_conv[conv_layer_num] = tf.nn.relu(tf.nn.bias_add(
                tf.nn.conv2d(input_conv, W_conv[conv_layer_num], strides=[1, 1, 1, 1], padding='SAME'),
                b_conv[conv_layer_num]))

        single_time_frame_net_variables = []
        single_time_frame_net_variables.extend(W_conv)
        single_time_frame_net_variables.extend(b_conv)
        self.savers['single_time_frame_net'] = tf.train.Saver(single_time_frame_net_variables)
        self.variables['single_time_frame_net'] = single_time_frame_net_variables

        h_conv_reshaped = tf.reshape(h_conv[-1],
                                     [-1, temporal_window_size * num_neurons_in_convolution_layers[-1]])
        W_conv_time = [None] * len(num_neurons_in_convolution_layers_for_time)
        b_conv_time = [None] * len(num_neurons_in_convolution_layers_for_time)
        h_conv_time = [None] * len(num_neurons_in_convolution_layers_for_time)

        conv_temp_window_size = temporal_window_size
        for conv_layer_num in range(len(num_neurons_in_convolution_layers_for_time)):
            conv_temp_window_size /= 2
            input_size = num_neurons_in_convolution_layers[-1] * 2
            if conv_layer_num == 0:
                input_conv_time = tf.reshape(h_conv_reshaped,
                                             [-1, 1, conv_temp_window_size, input_size])
            else:
                input_size = num_neurons_in_convolution_layers_for_time[conv_layer_num - 1] * 2
                input_conv_time = tf.reshape(h_conv_time[conv_layer_num - 1],
                                             [-1, 1, conv_temp_window_size, input_size])

            W_conv_time[conv_layer_num] = weight_variable([1, 1, input_size, num_neurons_in_convolution_layers_for_time[conv_layer_num]],
                                                          "convolution_time_" + str(conv_layer_num))
            b_conv_time[conv_layer_num] = bias_variable([num_neurons_in_convolution_layers_for_time[conv_layer_num]],
                                                        "convolution_time_" + str(conv_layer_num))
            h_conv_time[conv_layer_num] = tf.nn.relu(tf.nn.bias_add(
                tf.nn.conv2d(input_conv_time, W_conv_time[conv_layer_num], strides=[1, 1, 1, 1], padding='SAME'),
                b_conv_time[conv_layer_num]))

        two_time_frames_net_variables = []
        two_time_frames_net_variables.extend(W_conv_time)
        two_time_frames_net_variables.extend(b_conv_time)
        self.savers['two_time_frames_net'] = tf.train.Saver(two_time_frames_net_variables)
        self.variables['two_time_frames_net'] = two_time_frames_net_variables

        h_conv_time_reshaped = tf.reshape(h_conv_time[-1],
                                          [-1, conv_temp_window_size * num_neurons_in_convolution_layers_for_time[-1]])
        W_fc = [None] * len(num_neurons_in_fully_connected_layers)
        b_fc = [None] * len(num_neurons_in_fully_connected_layers)
        h_fc = [None] * len(num_neurons_in_fully_connected_layers)

        for layerNum in range(len(num_neurons_in_fully_connected_layers)):
            if layerNum == 0:
                fc_input = h_conv_time_reshaped
                fc_input_size = num_neurons_in_convolution_layers_for_time[-1] * conv_temp_window_size
            else:
                fc_input = h_fc[layerNum - 1]
                fc_input_size = num_neurons_in_fully_connected_layers[layerNum - 1]
            fc_output_size = num_neurons_in_fully_connected_layers[layerNum]

            W_fc[layerNum] = weight_variable([fc_input_size, fc_output_size], "fc_" + str(layerNum))
            b_fc[layerNum] = bias_variable([fc_output_size], "fc_" + str(layerNum))
            h_fc[layerNum] = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc_input, W_fc[layerNum]), b_fc[layerNum]))

        fully_connected_net_variables = []
        fully_connected_net_variables.extend(W_fc)
        fully_connected_net_variables.extend(b_fc)
        self.savers['fully_connected_net'] = tf.train.Saver(fully_connected_net_variables)
        self.variables['fully_connected_net'] = fully_connected_net_variables

        W_fc_last = weight_variable([num_neurons_in_fully_connected_layers[-1], num_actions], "fc_last")
        b_fc_last = bias_variable([num_actions], "fc_last")

        action_net_variables = []
        action_net_variables.append(W_fc_last)
        action_net_variables.append(b_fc_last)
        self.savers['action_net'] = tf.train.Saver(action_net_variables)
        self.variables['action_net'] = action_net_variables

        self.predicted_action_values = tf.nn.bias_add(tf.matmul(h_fc[-1], W_fc_last), b_fc_last)

        self.errors = tf.reduce_sum(tf.abs((self.y_ - self.predicted_action_values) * self.y_))

        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.errors, var_list=[].extend(self.variables['fully_connected_net']))
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

    def train(self, x_, y_, num_iterations, max_error, variables):
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
        variables = ['fully_connected_net']
        if variables is None:
            trainer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.errors)
        else:
            var_list = []
            for var_name in variables:
                var_list.extend(self.variables[var_name])

            trainer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.errors, var_list=var_list)

        for i in range(num_iterations):
            feed_dict = {self.x: x_, self.y_: y_}

            self.trainer.run(session=self.sess, feed_dict=feed_dict)
            error = self.sess.run(self.errors, feed_dict=feed_dict) / float(len(y_) / self.num_actions)
            print('\t\tloss: ' + str(error))
            if error < max_error:
                break

    def save_multi_file(self, model_base_name, extension):
        for saver_name in self.savers:
            file_name = model_base_name + '_' + saver_name + extension
            self.savers[saver_name].save(self.sess, file_name)

    def load_multi_file(self, model_base_name, extension):
        for saver_name in self.savers:
            file_name = model_base_name + '_' + saver_name + extension
            if os.path.exists(file_name):
                self.savers[saver_name].restore(self.sess, file_name)
