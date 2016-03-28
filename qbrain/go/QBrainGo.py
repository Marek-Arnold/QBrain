import os

from qbrain.go.QBrainGoNet import QBrainGoNet
from qbrain.go.QBrainGoMemory import QBrainGoMemory


class QBrainGo:
    """
    Class to handle reinforced experience based q-learning.
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
        :return: QBrain
        """
        self.field_size = board_size * board_size

        self.net = QBrainGoNet(
            board_size,
            convolution_layers,
            fully_connected_layers)

        self.mem = QBrainGoMemory(self.field_size, self.field_size + 1)

    def forward(self, group_name, input_features, possible_moves, time, is_black):
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
        :param: is_black: boolean

        Returns
        -------
        :return: int
            The index of the best action.
        """
        print('\tget_mem')
        running_experience = input_features

        if not is_black:
            inverted_exp = [i * -1.0 for i in running_experience]
            running_experience = inverted_exp

        # print memExp
        print('\tpredict')
        prediction = self.net.predict([running_experience], [possible_moves])[0]
        print(prediction)
        action = -1
        best_val = -100
        for i in range(0, self.field_size + 1):
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

    def train(self, batch_size, num_iter, max_error, train_layer_name):
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
        batch = self.mem.get_batch(batch_size, 0)
        if batch is None:
            print("Batch was none!")
            return
        print('\ttrain')
        self.net.train(batch[0], batch[1], num_iter, max_error, train_layer_name)

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
        print('saved')

    def load(self, model_base_name):
        self.net.load_multi_file('saves/' + model_base_name, '.ckpt')
        self.mem.load('saves/', model_base_name, '.pkl')
        print('loaded')

