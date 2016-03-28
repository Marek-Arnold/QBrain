class GoExperience:
    """
    Class to hold an observation, the taken action and the resulting reward.
    """
    def __init__(self, input_features, action, time):
        """
        Parameters
        ----------
        :param input_features: list of float
            Observed features of a single point in time.
        :param action: int
            The action that was taken based on this observation.
        :param time: int
            The time at which this observation was made.

        Returns
        -------
        :return: Experience
        """
        self.input_features = input_features
        self.time = time
        self.action = action
        self.reward = 0

    @staticmethod
    def get_empty_experience(num_inputs, time):
        """
        Use this method to get an Experience with no made observation and no taken action.

        Parameters
        ----------
        :param num_inputs: int
            The number of features that are normally observed.
        :param time: int
            The time at which this observation was made.

        Returns
        -------
        :return: Experience
            An empty experience.
        """
        return GoExperience([0] * num_inputs, -1, time)