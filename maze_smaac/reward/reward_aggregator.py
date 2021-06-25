"""Contains a default reward aggregator. This is Maze-specific and unrelated to the SMAAC approach."""
from abc import ABC
from abc import abstractmethod
from typing import List, Type

from maze_smaac.env.events import RewardEvents
from maze.core.env.reward import RewardAggregatorInterface


class BaseRewardAggregator(RewardAggregatorInterface):
    """Event aggregation object dealing with cutting rewards.
    """

    @abstractmethod
    def summarize_reward(self) -> float:
        """Summarize reward based on the orders and pieces to cut.

        :return: the summarized scalar reward.
        """
        raise NotImplementedError

    @classmethod
    def to_scalar_reward(cls, reward: float) -> float:
        """Nothing to do here for this env.

        :param: reward: already a scalar reward
        :return: the same scalar reward
        """
        return reward


class RewardAggregator(BaseRewardAggregator):
    """Default event aggregation object dealing with rewards.

    :param reward_scale: global reward scaling factor
    """

    def __init__(self, reward_scale: float):
        super().__init__()
        self.reward_scale = reward_scale

    def get_interfaces(self) -> List[Type[ABC]]:
        """Provides a list of reward relevant event interfaces.

        :return: List of event interfaces.
        """""
        return [RewardEvents]

    def summarize_reward(self) -> float:
        """Summarizes all reward relevant events to a scalar value.

        :return: The accumulated scalar reward.
        """
        total_reward = 0.0

        # process the l2rpn environment reward
        l2rpn_rewards = [ev for ev in self.query_events([RewardEvents.l2rpn_reward])]
        assert len(l2rpn_rewards) == 1
        total_reward += l2rpn_rewards[0].reward

        return total_reward * self.reward_scale
