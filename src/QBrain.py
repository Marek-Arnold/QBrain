from QBrainNet import QBrainNet
from QBrainMemory import QBrainMemory
import os


class QBrain:
    """
    Class to handle reinforced experience based q-learning.
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
            Size of a single observation together with the one hot encoded action (num features + num actions).
        :param temporal_window_size: int
            Number of observations to use for each prediction.
        :param num_actions: int
            The number of possible actions that can be taken.
        :param sensor_descriptions: list of tuple of int and int and list of int
            List of tuples that describe the sensors [(num_sensors, num_inputs_per_sensor, network_per_sensor, network_for_sensors)]
            num_sensors: Number of sensors of this kind.
            num_inputs_per_sensor: Number of features one sensor observes.
            network_per_sensor: List of int to describe the network to use for the processing of this single sensor.
            network_for_sensors: List of int to describe the network to use for all sensors of this kind in one time step.
        :param num_neurons_in_convolution_layers: list of int
            Number of features in the convolution layers that should be learned for the representation of a single input.
        :param num_neurons_in_convolution_layers_for_time: list of int
            Number of features in the convolution layers that should be learned for the representation of two single inputs.
        :param num_neurons_in_fully_connected_layers: list of int
            Number of neurons in the fully connected layers.

        Returns
        -------
        :return: QBrain
        """
        self.temporal_window_size = temporal_window_size
        self.num_actions = num_actions

        self.net = QBrainNet(single_input_size,
                             temporal_window_size,
                             num_actions,
                             sensor_descriptions,
                             num_neurons_in_convolution_layers,
                             num_neurons_in_convolution_layers_for_time,
                             num_neurons_in_fully_connected_layers)

        self.mem = QBrainMemory(single_input_size, num_actions)

    def forward(self, group_name, input_features, time):
        """
        Predict the q-values for the given group and observation.

        Parameters
        ----------
        :param group_name: str
            The name of the group.
        :param input_features: list of float
            The features of the actual observation.
        :param time: int
            The time of this observation.

        Returns
        -------
        :return: int
            The index of the best action.
        """
        print('\tget_mem')
        running_experience = self.mem.get_experiences_for_prediction(group_name, input_features, time, self.temporal_window_size)
        # print memExp
        print('\tpredict')
        prediction = self.net.predict([running_experience])[0]
        print(prediction)
        action = -1
        best_val = -100
        for i in range(0, self.num_actions):
            if prediction[i] > best_val:
                action = i
                best_val = prediction[i]
        self.mem.put_experience(group_name, input_features, action, time)
        return action

    def expert_forward(self, group_name, input_features, action, time):
        """
        Feed an expert decision into the observations.

        Parameters
        ----------
        :param group_name: str
            The name of the group.
        :param input_features: list of float
            The features of the actual observation.
        :param action: int
            The action taken by the expert.
        :param time:
            The time of this expert action.

        Returns
        -------
        :return: None
        """
        self.mem.put_experience(group_name, input_features, action, time)

    def train(self, batch_size, num_iter):
        """
        Train based on flushed groups experiences.

        Parameters
        ----------
        :param batch_size: int
            The number of series of observations that should be used for training.
        :param num_iter: int
            The number of iterations to train on this batch.

        Returns
        -------
        :return: None
        """
        print('\tget_batch')
        batch = self.mem.get_batch(batch_size, self.temporal_window_size)
        if batch is None:
            print("Batch was none!")
            return
        print('\ttrain')
        self.net.train(batch[0], batch[1], num_iter)

    def post_reward(self, group_name, reward, start_time, duration):
        """
        Put a reward in the memory.

        Parameters
        ----------
        :param group_name: str
            The name of the group to reward.
        :param reward: float
            The amount of the reward that should be distributed.
        :param start_time:
            The first time slot that should be affected by this reward.
        :param duration:
            The duration over which this reward is distributed.

        Returns
        -------
        :return: None
        """
        self.mem.put_reward(group_name, reward, start_time, duration)

    def flush_group(self, group_name):
        """
        Flush the given group. This will distribute all rewards and make this observations ready for training.

        Parameters
        ----------
        :param group_name: str
            The group to flush.

        Returns
        -------
        :return: None
        """
        self.mem.flush_group(group_name)

    def save(self, model_name):
        """
        Persist the model and the experience.

        Parameters
        ----------
        :param model_name: str
            The base name to use for files.

        Returns
        -------
        :return: None
        """
        if not os.path.exists('saves/'):
            os.mkdir('saves/')
        self.mem.save('saves/', model_name, '.pkl')
        self.net.save_multi_file('saves/' + model_name, '.ckpt')

    def load(self, model_base_name):
        self.net.load_multi_file('saves/' + model_base_name, '.ckpt')
        self.mem.load('saves/', model_base_name, '.pkl')

