""" Contains a planner for the low-level actions used by SMAAC. Adopted mainly from
 https://github.com/KAIST-AILab/SMAAC/blob/53c52f35dfa9224d1adfc5d3e9e67912b7cf0f1b/agent.py """

from typing import List, Optional, Tuple, Dict

import grid2op
import numpy as np
import torch
from grid2op.Observation import CompleteObservation

from maze_smaac.space_interfaces.dict_action_conversion import MazeSMAACActionConversion
from maze_smaac.space_interfaces.dict_observation_conversion import MazeSMAACObservationConversion


class Planner:
    """ Plans low-level actions for a given high-level action ("goal")

    :param action_conversion: A Maze ActionConversion
    :param observation_conversion: A Maze ObservationConversion
    """
    def __init__(self, action_conversion: MazeSMAACActionConversion, observation_conversion: MazeSMAACObservationConversion,
                 rule: str, max_low_len_step_count: int):
        self.observation_conversion = observation_conversion
        self.action_conversion = action_conversion
        self.rule = rule
        self.max_low_len_step_count = max_low_len_step_count

    def reconnect_line(self, maze_state: grid2op.Observation.CompleteObservation) ->\
            Optional[Dict[str, Dict]]:
        """Generate power line reconnection maze_action from maze_state.
        (Only one line is reconnected at a time)

        :param maze_state: The grid2op state for which reconnect actions are to be generated.
        """

        # indices of disconnected lines
        dislines = np.where(maze_state.line_status is False)[0]

        # iterate disconnected lines
        for i in dislines:
            if maze_state.time_next_maintenance[i] != 0 and i in self.action_conversion.lonely_lines:
                sub_or = self.action_conversion.action_space.line_or_to_subid[i]
                sub_ex = self.action_conversion.action_space.line_ex_to_subid[i]

                if maze_state.time_before_cooldown_sub[sub_or] == 0:
                    return {'set_bus': {'lines_or_id': [(i, 1)]}}

                if maze_state.time_before_cooldown_sub[sub_ex] == 0:
                    return {'set_bus': {'lines_ex_id': [(i, 1)]}}

                if maze_state.time_before_cooldown_line[i] == 0:
                    status = self.action_conversion.action_space.get_change_line_status_vect()
                    status[i] = True
                    return {'change_line_status': status}

        return None

    def compute_low_level_actions_toward_goal(self, bus_goal: np.ndarray, order: Optional,
                                              maze_state: CompleteObservation) -> List[Tuple[int, np.ndarray, int]]:
        """ Computes low level actions for a given bus configuration (the bus_goal)

        :param bus_goal: The target bus configuration.
        :param maze_state: The MazeState that will be used for low action optimization.
        """
        # compute low level actions towards this goal topology
        low_actions = self.plan_act(goal=bus_goal, maze_state=maze_state, sub_order_score=order)
        low_actions = self.optimize_low_actions(maze_state=maze_state, low_actions=low_actions)

        return low_actions

    def optimize_low_actions(self, maze_state: CompleteObservation, low_actions: List[Tuple[int, np.ndarray, int]]) -> \
            List[Tuple[int, np.ndarray, int]]:
        """Optimizes low level actions considering substation cool down times.
        If optimizing it is not feasible (current action has cool down) discard the action sequence entirely.

        :param maze_state: The MazeState for which the low actions should be optimized.
        :param low_actions: A list of low actions.
        """

        # remove overlapped action
        optimized = []
        cool_down_list = maze_state.time_before_cooldown_sub
        if self.max_low_len_step_count != 1 and self.rule == 'c':
            low_actions = self.heuristic_order(maze_state, low_actions)

        for low_act in low_actions:
            sub_id, sub_goal = low_act[:2]  # substation id, substation bus
            sub_goal, same = self.inspect_act(sub_id, sub_goal, maze_state,
                                              subs=self.observation_conversion.subs)
            if not same:
                optimized.append((sub_id, sub_goal, cool_down_list[sub_id]))

        # sort by cooldown_sub
        if self.max_low_len_step_count != 1 and self.rule != 'o':
            optimized = sorted(optimized, key=lambda x: x[2])

        # if current action has cool down, then discard
        if len(optimized) > 0 and optimized[0][2] > 0:
            optimized = []

        return optimized

    def plan_act(self, goal: np.ndarray, maze_state: grid2op.Observation.CompleteObservation,
                 sub_order_score: Optional[torch.Tensor] = None) -> List[Tuple[int, np.ndarray]]:
        """Plan required sub-steps towards goal topology.

        :param goal: The current bus goal.
        :param maze_state: MazeState from which to plan actions.
        :param sub_order_score: Not used in this implementation (it's always None, see assertion below)
        """
        assert sub_order_score is None, "sub_order_score must be None in this implementation"

        # get current topology vector
        topo_vect = maze_state.topo_vect[self.action_conversion.sub_mask]

        # initialize sub-steps towards goal topology
        targets = []

        # convert goal and shift to bus IDs [0, 1] -> [1, 2]
        assert goal.ndim == 1
        goal += 1

        # process order of substations
        if sub_order_score is None:
            sub_order = self.action_conversion.masked_sorted_sub
        else:
            sub_order = [i[0] for i in sorted(list(zip(self.action_conversion.masked_sorted_sub,
                                                       sub_order_score[0].tolist())),
                                              key=lambda x: -x[1])]

        # iterate substations
        for sub_id in sub_order:

            # extract current substation topo sub-vector
            beg = self.action_conversion.masked_sub_to_topo_begin[sub_id]
            end = self.action_conversion.masked_sub_to_topo_end[sub_id]

            # current topology
            topo = topo_vect[beg:end]
            # new topology for
            new_topo = goal[beg:end]

            # if current and new topology diverge add to targets (sub-steps)
            if np.any(new_topo != topo).item():
                targets.append((sub_id, new_topo))

        # prepare plan (action sequence towards the goal)
        plan = [(sub_id, new_topo) for sub_id, new_topo in targets]
        return plan

    def inspect_act(self, sub_id: int, goal: np.ndarray, maze_state: CompleteObservation,
                    subs: List[Dict[str, List[int]]]) -> Tuple[np.ndarray, bool]:
        """ Corrects illegal actions

        :param sub_id: Substation ID, for which to correct actions.
        :param goal: High-level goal action.
        :param maze_state: MazeState for which to inspect actions.
        :param subs: Dict of substations.
        """
        # Correct illegal action
        # collect original ids
        exs = subs[sub_id]['e']
        ors = subs[sub_id]['o']
        lines = exs + ors  # [line_id0, line_id1, line_id2, ...]

        # just prevent isolation
        line_idx = len(lines) - 1
        if (goal[:line_idx] == 1).all() * (goal[line_idx:] != 1).any():
            goal = np.ones_like(goal)

        if torch.is_tensor(goal):
            goal = goal.numpy()
        beg = self.action_conversion.masked_sub_to_topo_begin[sub_id]
        end = self.action_conversion.masked_sub_to_topo_end[sub_id]

        # check if target topology is already reached
        already_same = np.all(goal == maze_state.topo_vect[self.action_conversion.sub_mask][beg:end])

        return goal, already_same

    def heuristic_order(self, maze_state: grid2op.Observation.CompleteObservation,
                        low_actions: List[Tuple[int, np.ndarray, int]]) -> List[Tuple[int, np.ndarray, int]]:
        """ Taken from ORIGINAL_REPO.converter.graphGoalConverter.heuristic_order

        :param maze_state: The MazeState for which the low actions should be optimized.
        :param low_actions: A list of low actions.
        :return: A potentially re-ordered list of low actions.
        """
        if len(low_actions) == 0:
            return []
        rhos = []
        for item in low_actions:
            sub_id = item[0]
            lines = self.observation_conversion.subs[sub_id]['e'] + self.observation_conversion.subs[sub_id]['o']
            rho = maze_state.rho[lines].copy()
            rho[rho == 0] = 3
            rho_max = rho.max()
            rho_mean = rho.mean()
            rhos.append((rho_max, rho_mean))
        order = sorted(zip(low_actions, rhos), key=lambda x: (-x[1][0], -x[1][1]))
        return list(list(zip(*order))[0])
