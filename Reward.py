class Reward:
    """
    Class to store a reward.
    """
    def __init__(self, reward, start_time, duration):
        """
        Parameters
        ----------
        :param reward: float
            The total reward.
        :param start_time: int
            The first time slot this reward applies to.
        :param duration: int
            The number of time slots this reward applies to.

        Returns
        -------
        :return: Reward
        """
        self.reward = reward
        self.start_time = start_time
        self.duration = duration