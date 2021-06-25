"""Contains events for the l2rpn environment. This is Maze-specific and unrelated to the SMAAC approach."""
from abc import ABC

import numpy as np
from maze.core.log_stats.event_decorators import define_step_stats, define_episode_stats, \
    define_stats_grouping, define_epoch_stats


class ActionEvents(ABC):
    """Event related to actions made by an actor"""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def illegal_action_performed(self, redispatch: bool, reconnect: bool):
        """The event of a agent making a illegal action

        :param redispatch: illegal due to redispatching
        :param reconnect: illegal due to a power-line reconnection
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def ambiguous_action_performed(self):
        """The event of a agent making a ambiguous action
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def noop_action_performed(self):
        """The event of a agent having no effect on the grid
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def topology_action_performed(self):
        """The event of a agent performing a topology change
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def redispatch_action_performed(self):
        """The event of a agent performing a redispatching
        """


class GridEvents(ABC):
    """Event related to a power-grid"""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    @define_stats_grouping("exception")
    def done(self, exception: str):
        """The event is fired on every done episode, logging the causing exception (or "none")
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    @define_stats_grouping("exception")
    def not_done_exception(self, exception: str):
        """The event is fired whenever an Exception is fired which does not cause a done=True
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def power_line_overload(self, line_id: int, rho: float):
        """The event of a given power-line overloading

        :param line_id: power line that is currently overflowing
        :param rho: The capacity of the powerline. It is defined at the observed current flow divided by the thermal
        limit of the powerline (no unit)
        """


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


class SMDPEvents(ABC):
    """ Event related to Semi-MPDs """

    @define_epoch_stats(np.mean, output_name="internal_step")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def internal_step(self):
        """ The event of performing an internal step"""

    @define_epoch_stats(np.mean, output_name="new_goal_required")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def new_goal_required(self):
        """ Event of requiring a new high-level action"""

    @define_epoch_stats(np.mean, output_name="no_low_level_action_left")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def no_low_level_action_left(self):
        """ Event if low_level list inside SMPDWrapper is empty """

    @define_epoch_stats(np.mean, output_name="regular_noop")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def regular_noop(self):
        """ Event of a noop being caused in the SMPDWrapper """

    @define_epoch_stats(np.mean, output_name="noop_in_reset")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def noop_in_reset(self):
        """ Event of a noop being performed in the reset method. """

    @define_epoch_stats(np.mean, output_name="regular_action")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def regular_action(self):
        """ Event of a regular action being caused in the SMPDWrapper """

    @define_epoch_stats(np.mean, output_name="done_in_smdp_wrapper")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def done_in_smdp_wrapper(self):
        """ Event of a done flag being encountered in the SMPDWrapper """
