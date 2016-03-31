import random
import os
import os.path
import pickle

from qbrain.core.Experience import Experience
from qbrain.core.ExperienceGroup import ExperienceGroup
from qbrain.core.Reward import Reward


class QBrainGoMemory:
    """
    Memory to work with and to persist Experience.
    """

    def __init__(self, num_inputs, num_actions, path, base_name, extension):
        """

        Parameters
        ----------
        :param num_inputs: int
            The number of inputs for a single observation.
        :param num_actions:
            The number of actions that can be taken.

        Returns
        -------
        :return: QBrainMemory
        """
        self.path = path
        self.base_name = base_name
        self.extension = extension

        self.experience_groups = {}
        self.reward_groups = {}
        self.flushed_experience_groups = {}
        self.num_actions = num_actions
        self.num_inputs = num_inputs

    def put_experience(self, group_name, input_features, action, time):
        """
        Put an experience in the memory.

        Parameters
        ----------
        :param group_name: str
            The name of the group to add an experience for.
        :param input_features:
            The made observation.
        :param action:
            The action that was taken based on this observation.
        :param time:
            The time at which this observation was made.

        Returns
        -------
        :return: None
        """
        if group_name not in self.experience_groups:
            self.experience_groups[group_name] = ExperienceGroup()

        self.experience_groups[group_name].add(Experience(input_features=input_features, action=action, time=time))

    def put_reward(self, group_name, reward, start_time, duration):
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
        if group_name not in self.reward_groups:
            self.reward_groups[group_name] = []

        reward_group = self.reward_groups[group_name]
        reward_group.append(Reward(reward, start_time, duration))

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
        if group_name not in self.experience_groups or group_name not in self.reward_groups:
            print("WARNING - group_name unknown: " + group_name)
        else:
            experience_group = self.experience_groups[group_name]
            for reward in self.reward_groups[group_name]:
                r = reward.reward / reward.duration
                for time in range(reward.start_time, reward.start_time + reward.duration):
                    if time in experience_group.group:
                        experience = experience_group.group[time]
                        experience.reward += r

            self.save_group(experience_group, group_name)
            self.flushed_experience_groups[group_name] = None
            self.experience_groups.pop(group_name)
            self.reward_groups.pop(group_name)

    def get_save_path(self, group_name):
        full_path = self.path + self.base_name + '_' + group_name + self.extension
        return full_path

    def save_group(self, group, group_name):
        full_path = self.get_save_path(group_name)
        if os.path.isfile(full_path):
            print('Not overwriting ' + full_path + ' as it already exists.')
        else:
            self.save_obj(group, full_path)

    def load_group(self, group_name):
        full_path = self.get_save_path(group_name)
        if not os.path.exists(full_path):
            print('Group not found! ' + full_path)
            return None
        else:
            return self.load_obj(full_path)

    def get_batch(self, batch_size):
        """
        Get a batch for training. From all flushed groups random observation series are taken for each batch and filled
        with empty experiences where needed.

        Parameters
        ----------
        :param batch_size: int
            The desired batch size.

        Returns
        -------
        :return: tuple of list of list of float and list of list of float
        """
        if len(self.flushed_experience_groups) == 0:
            return None

        return self.get_batch_from_groups(batch_size, self.flushed_experience_groups, self.num_actions)

    def load(self):
        """
        Restore some previously saved flushed groups.

        Parameters
        ----------
        :param path: str
            The path for the files where the flushed groups are stored.
        :param base_name: str
            The first part of the name for the flushed groups.
        :param extension: str
            The extension to use for the files.

        Returns
        -------
        :return: None
        """
        self.flushed_experience_groups = {}
        if not os.path.exists(self.path):
            print('Path not found! ' + self.path)
        else:
            for file_name in os.listdir(self.path):
                if file_name[:len(self.base_name)] == self.base_name and file_name[-len(self.extension):] == self.extension:
                    self.flushed_experience_groups[file_name[len(self.base_name) + 1:-len(self.extension)]] = None

    def get_batch_from_groups(self, batch_size, groups, num_actions):
        batch_x = []
        batch_y = []
        for batch_num in range(0, batch_size):
            group = self.load_group(random.choice(list(groups.keys())))
            ind = random.randint(group.first, group.last)
            experience = group.group[ind]
            batch_x.append(experience.input_features)
            batch_y.append(QBrainGoMemory.get_y_from_experience(experience, num_actions))

        return batch_x, batch_y

    @staticmethod
    def get_one_hot_y_from_experience(experience, num_actions):
        y = []
        for i in range(0, num_actions):
            if i == experience.action:
                y.append(1)
            else:
                y.append(0)
        return y

    @staticmethod
    def get_y_from_experience(experience, num_actions):
        y = []
        for i in range(0, num_actions):
            if i == experience.action:
                y.append(experience.reward)
            else:
                y.append(0)
        return y

    @staticmethod
    def save_obj(obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name):
        with open(name, 'rb') as f:
            return pickle.load(f)







