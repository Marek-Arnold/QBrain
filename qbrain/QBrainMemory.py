from qbrain import Experience
from qbrain import ExperienceGroup
from qbrain import Reward
import random
import os
import os.path
import pickle


class QBrainMemory:
    """
    Memory to work with and to persist Experience.
    """
    def __init__(self, num_inputs, num_actions):
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
        self.experience_groups = {}
        self.reward_groups = {}
        self.flushed_experience_groups = {}
        self.num_actions = num_actions
        self.num_inputs = num_inputs

    def get_experiences_for_prediction(self, group_name, input_features, time, temporal_window):
        """
        Get a series of observations in the form |input_features20|one_hot_action19|input_features19|one_hot_action18|..

        Parameters
        ----------
        :param group_name: str
            The name of the group to get the observations for.
        :param input_features: list of float
            The newest observation to prepend to all others.
        :param time: int
            The time of the newest observation.
        :param temporal_window: int
            The amount of observations to combine to a series.

        Returns
        -------
        :return: list of float
        """
        experiences = []
        if group_name not in self.experience_groups:
            self.experience_groups[group_name] = ExperienceGroup.ExperienceGroup()

        group = self.experience_groups[group_name]

        for i in range(time - temporal_window, time):
            if i in group.group:
                experiences.append(group.group[i])
            else:
                experiences.append(Experience.Experience.get_empty_experience(self.num_inputs - self.num_actions, i))
        experiences.append(Experience.Experience(input_features, -1, time))
        experiences.reverse()
        b = QBrainMemory.get_xy_from_experiences(experiences, self.num_actions)
        return b[0]

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
            self.experience_groups[group_name] = ExperienceGroup.ExperienceGroup()

        self.experience_groups[group_name].add(Experience.Experience(input_features=input_features, action=action, time=time))

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
        reward_group.append(Reward.Reward(reward, start_time, duration))

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

            self.flushed_experience_groups[group_name] = experience_group
            self.experience_groups.pop(group_name)
            self.reward_groups.pop(group_name)

    def get_batch(self, batch_size, temporal_window):
        """
        Get a batch for training. From all flushed groups random observation series are taken for each batch and filled
        with empty experiences where needed.

        Parameters
        ----------
        :param batch_size: int
            The desired batch size.
        :param temporal_window:
            The number of observations per series.

        Returns
        -------
        :return: tuple of list of list of float and list of list of float
        """
        if len(self.flushed_experience_groups) == 0:
            return None

        return self.get_batch_from_groups(batch_size, temporal_window, self.flushed_experience_groups, self.num_actions, self.num_inputs)

    def save(self, path, base_name, extension):
        """
        Save all flushed groups to disk.

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

        for group_name in self.flushed_experience_groups:
            full_path = path + base_name + '_' + group_name + extension
            if os.path.isfile(full_path):
                print('Not overwriting ' + full_path + ' as it already exists.')
            else:
                self.save_obj(self.flushed_experience_groups[group_name], full_path)

    def load(self, path, base_name, extension):
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
        if not os.path.exists(path):
            print('Path not found! ' + path)
        else:
            for file_name in os.listdir(path):
                if file_name[:len(base_name)] == base_name and file_name[-len(extension):] == extension:
                    self.flushed_experience_groups[file_name[len(base_name) + 1:-len(extension)]] = self.load_obj(path + file_name)

    def load_single_file(self, name):
        """
        Restore some previously saved flushed groups.

        Parameters
        ----------
        :param name: str
            The name and path of the flushed experience groups.

        Returns
        -------
        :return: None
        """
        self.flushed_experience_groups = self.load_obj(name)

    @staticmethod
    def get_batch_from_groups(batch_size, temporal_window, groups, num_actions, num_inputs):
        batch_x = []
        batch_y = []
        for batch_num in range(0, batch_size):
            experiences = []
            group = random.choice(groups.values())
            ind = random.randint(group.first, group.last)

            for i in range(ind - temporal_window, ind + 1):
                if i in group.group:
                    experiences.append(group.group[i])
                else:
                    experiences.append(Experience.Experience.get_empty_experience(num_inputs - num_actions, i))
            experiences.reverse()
            b = QBrainMemory.get_xy_from_experiences(experiences, num_actions)
            batch_x.append(b[0])
            batch_y.append(b[1])

        return batch_x, batch_y

    @staticmethod
    def get_xy_from_experiences(experiences, num_actions):
        x = []

        x.extend(experiences[0].input_features)
        for i in range(1, len(experiences) - 1):
            x.extend(QBrainMemory.get_one_hot_y_from_experience(experiences[i], num_actions))
            x.extend(experiences[i].input_features)

        x.extend(QBrainMemory.get_one_hot_y_from_experience(experiences[-1], num_actions))
        y = QBrainMemory.get_y_from_experience(experiences[0], num_actions)

        return x,y

    @staticmethod
    def get_one_hot_y_from_experience(experience, num_actions):
        y = []
        for i in range(0,num_actions):
            if i == experience.action:
                y.append(1)
            else:
                y.append(0)
        return y

    @staticmethod
    def get_y_from_experience(experience, num_actions):
        y = []
        for i in range(0,num_actions):
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







