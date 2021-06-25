"""Contains a Maze ActionConversion interface. Contains code adopted from
https://github.com/KAIST-AILab/SMAAC/blob/53c52f35dfa9224d1adfc5d3e9e67912b7cf0f1b/converter.py """
from typing import Dict

import grid2op
import numpy as np
from grid2op.Action import TopologyAndDispatchAction
from gym import spaces
from maze.core.env.action_conversion import ActionConversionInterface, ActionType
from maze.core.env.maze_state import MazeStateType


class MazeSMAACActionConversion(ActionConversionInterface):
    """
    L2RPN MazeState to dictionary ActionConversion

    :param env: A grid2op Environment.
    :param mask: Lower substation mask parameter.
    :param mask_hi: Upper substation mask parameter.
    :param danger: The danger threshold above which to act. delta_t in the paper.
    :param bus_threshold: Corresponds to delta_t in Yoon et al. (2021) - Winning the l2rpn challenge:
           power grid management via semi markov afterstate actor critic.
    """

    def __init__(self, env: grid2op.Environment.Environment, mask: int, mask_hi: int, danger: float,
                 bus_threshold: float):
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        self.mask = mask
        self.mask_hi = mask_hi
        self.danger = danger
        self.bus_threshold = bus_threshold

        self.thermal_limit_under400 = env._thermal_limit_a < 400

        # split topology over sub_id
        self.sub_to_topo_begin, self.sub_to_topo_end = [], []
        idx = 0
        for num_topo in self.action_space.sub_info:
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)

        self._init_action_converter()

    def space(self) -> spaces.Dict:
        """ Returns gym action space """
        action_space_dict = dict()

        action_space_dict['goal_topology'] = spaces.Box(low=-np.float(1), high=np.float(1), shape=(self.n,))

        return spaces.Dict(action_space_dict)

    def space_to_maze(self, action: ActionType, maze_state: MazeStateType) -> TopologyAndDispatchAction:
        """ Transforms agent action to L2RPN MazeAction

        :param action: The agent action.
        :param maze_state: Current MazeState.
        :return: Concert agent action to environment (simulation) action.
        """
        sub_id, new_topo = action
        if sub_id is None and new_topo is None:
            return self.action_space()

        new_topo = [1] + new_topo.tolist()
        grid2op_action = self.action_space({'set_bus': {'substations_id': [(sub_id, new_topo)]}})
        return grid2op_action

    def goal_action_to_bus_action(self, goal_action_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """ Maps goal action logits returned by model to bus action.

        :param goal_action_dict: Dict of bus goal action logits.
        """
        goal_action = goal_action_dict['goal_topology']
        # map prediction to bus goal
        bus_goal = np.zeros_like(goal_action, dtype=np.int)
        bus_goal[goal_action > self.bus_threshold] = 1  # why is self.bus_threshold = 0.35?

        return bus_goal

    def _init_action_converter(self) -> None:
        """Initialize action conversion"""
        self.sorted_sub = list(range(self.action_space.n_sub))
        self.sub_mask = []  # mask for parsing actionable topology
        self.psubs = []  # actionable substation IDs
        self.masked_sub_to_topo_begin = []
        self.masked_sub_to_topo_end = []
        idx = 0
        for i, num_topo in enumerate(self.action_space.sub_info):
            if num_topo > self.mask and num_topo < self.mask_hi:
                self.sub_mask.extend(
                    [j for j in range(self.sub_to_topo_begin[i] + 1, self.sub_to_topo_end[i])])
                self.psubs.append(i)
                self.masked_sub_to_topo_begin.append(idx)
                idx += num_topo - 1
                self.masked_sub_to_topo_end.append(idx)

            else:  # dummy
                self.masked_sub_to_topo_begin.append(-1)
                self.masked_sub_to_topo_end.append(-1)
        self.n = len(self.sub_mask)

        if self.obs_space.n_sub == 5:
            self.masked_sorted_sub = [0, 3, 2, 1, 4]
        elif self.obs_space.n_sub == 14:
            self.masked_sorted_sub = [5, 1, 3, 4, 2, 12, 0, 11, 13, 10, 9, 6, 7]
        elif self.obs_space.n_sub == 36:  # mask = 5
            self.masked_sorted_sub = [16, 23, 21, 26, 33, 29, 35, 9, 7, 4, 1]
            if self.mask == 4:
                self.masked_sorted_sub += [22, 27, 28, 32, 13]

        self.lonely_lines = set()
        for i in range(self.obs_space.n_line):
            if (self.obs_space.line_or_to_subid[i] not in self.psubs) \
                    and (self.obs_space.line_ex_to_subid[i] not in self.psubs):
                self.lonely_lines.add(i)
        self.lonely_lines = list(self.lonely_lines)
