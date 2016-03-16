import sys


class ExperienceGroup:
    """
    A group of Experiences.
    """
    def __init__(self):
        """
        :return: ExperienceGroup
            An empty group.
        """
        self.group = {}
        self.first = sys.maxsize
        self.last = -1

    def add(self, experience):
        """
        Add an experience to this group.

        Parameters
        ----------
        :param experience: Experience
            The Experience to add to this group.
        Returns
        -------
        :return: None
        """
        self.group[experience.time] = experience
        self.first = min(self.first, experience.time)
        self.last = max(self.last, experience.time)