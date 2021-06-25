"""Contains reward events for the l2rpn environment. This is Maze-specific and unrelated to the SMAAC approach. """
from abc import ABC


class RewardEvents(ABC):
    """Event related to grid2op rewards

    Grid2Op grants us access to multiple reward signals by accessing the info dict. These rewards will be forwarded as
    events!
    """

    def l2rpn_reward(self, reward: float):
        """The event of the original l2rpn environment reward

        :param reward: the value
        """

    def other_reward(self, name: str, reward: float, is_kpi: bool):
        """The event of a given reward with a given value

        :param name: reward name
        :param reward: the value
        :param is_kpi: if this score should be included in the KPIs
        """
