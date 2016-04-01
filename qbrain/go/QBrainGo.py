import os

from qbrain.go.Go import Go
from qbrain.go.QBrainGoNet import QBrainGoNet
from qbrain.go.QBrainGoMemory import QBrainGoMemory


class QBrainGo:
    """
    Class to handle reinforced experience based q-learning.
    """

    def __init__(self,
                 board_size,
                 convolution_layers,
                 fully_connected_layers,
                 mem_path, mem_base_name, mem_extension,
                 net_path, net_base_name, net_extension):
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
        self.mem_path = mem_path
        self.mem_base_name = mem_base_name
        self.mem_extension = mem_extension
        self.net_path = net_path
        self.net_base_name = net_base_name
        self.net_extension = net_extension

        self.field_size = board_size * board_size

        self.net = QBrainGoNet(
            board_size,
            convolution_layers,
            fully_connected_layers)

        self.mem = QBrainGoMemory(self.field_size, self.field_size + 1, self.mem_path, self.mem_base_name, self.mem_extension)

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
            for x in range(len(running_experience)):
                for y in range(len(running_experience[x])):
                    r = running_experience[x][y]
                    if r == Go.black_field:
                        running_experience[x][y] = Go.white_field
                    elif r == Go.white_field:
                        running_experience[x][y] = Go.black_field

        # print memExp
        print('\tpredict')
        predicted_lower_bounds, predicted_upper_bounds = self.net.predict([running_experience], [possible_moves])
        # print(prediction)

        lower_bound_action = -1
        lower_bound_best_val = -1000000
        for i in range(0, self.field_size + 1):
            if (lower_bound_action == -1 or predicted_lower_bounds[i] > lower_bound_best_val) and possible_moves[i] != 0:
                lower_bound_action = i
                lower_bound_best_val = predicted_lower_bounds[i]

        if lower_bound_action == -1:
            action = self.field_size
        elif lower_bound_best_val < 0:
            upper_bound_action = -1
            upper_bound_best_val = -1000000
            for i in range(0, self.field_size + 1):
                if (upper_bound_action == -1 or predicted_upper_bounds[i] > upper_bound_best_val) and possible_moves[i] != 0:
                    upper_bound_action = i
                    upper_bound_best_val = predicted_upper_bounds[i]
            action = upper_bound_action
        else:
            action = lower_bound_action

        self.mem.put_experience(group_name, input_features, action, time)
        return action, predicted_lower_bounds[action], predicted_upper_bounds[action]

    def expert_forward(self, group_name, input_features, action, time, is_black):
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
        if not is_black:
            inverted_exp = [i * -1.0 for i in input_features]
            input_features = inverted_exp
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
        batch = self.mem.get_batch(batch_size)
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

    def save(self):
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
        if not os.path.exists(self.net_path):
            os.mkdir(self.net_path)
        self.net.save_multi_file(self.net_path + self.net_base_name, self.net_extension)
        print('saved')

    def load(self):
        self.net.load_multi_file(self.net_path + self.net_base_name, self.net_extension)
        self.mem.load()
        print('loaded')

