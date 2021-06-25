"""Contains the SMDP wrapper implementation to enable SMAAC. Adopted from
https://github.com/KAIST-AILab/SMAAC/blob/53c52f35dfa9224d1adfc5d3e9e67912b7cf0f1b/train.py """
from typing import Dict, List, Any, Tuple

import numpy as np
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.wrappers.wrapper import Wrapper

from maze_smaac.env.events import SMDPEvents
from maze_smaac.policies.planners import Planner


class MazeSmaacSMDPWrapper(Wrapper[MazeEnv]):
    """ Wrapper that controls the SMDP logic for the L2RPN challenge.

    :param env: A MazeEnv
    :param max_low_level_step_count: Parameter necessary to instantiate planner.
    :param smaac_evaluation_mode: If true, sets env to done after 864 steps (like the SMAAC authors do)
    :param verbose: If True, prints more output to the command line.
    """

    def __init__(self, env: MazeEnv, rule: str, max_low_level_step_count: int, smaac_evaluation_mode: bool,
                 verbose: bool = False):
        super(MazeSmaacSMDPWrapper, self).__init__(env=env, keep_inner_hooks=True)
        self.rule = rule
        self.max_low_level_step_count = max_low_level_step_count
        self.smaac_evaluation_mode = smaac_evaluation_mode
        self.verbose = verbose

        self.goal = None
        self.low_level_step_count = -1
        self.adj = None
        self.stacked_obs = []
        self.low_actions = []
        self.order = None
        self.bus_goal = None
        self.internal_steps = 0
        self.internal_rewards = []
        self.total_reward = 0
        self.topo = None
        self.after_reset = False
        self.total_steps_since_reset = 0

        # The planner will be responsible for calculating the low-level actions.
        self.low_level_policy = Planner(action_conversion=env.action_conversion,
                                        observation_conversion=env.observation_conversion, rule=rule,
                                        max_low_len_step_count=max_low_level_step_count)

        # create event topics
        self.smdp_events = self.core_env.context.event_service.create_event_topic(SMDPEvents)

    def reset(self) -> Any:
        """ Resets member variables and calls self.env.reset() """
        self.goal = None
        self.low_level_step_count = -1
        self.adj = None
        self.stacked_obs = []
        self.low_actions = []
        self.topo = None
        self.after_reset = True
        self.total_steps_since_reset = 0

        obs = self.env.reset()

        return obs

    def update_goal_member_variables(self, goal: Dict[str, np.ndarray], bus_goal: np.ndarray, low_actions: List,
                                     order=None) -> None:
        """Helper function to update the goal member variables."""
        self.order = order
        self.goal = goal
        self.bus_goal = bus_goal
        self.low_actions = low_actions
        self.low_level_step_count = 0

    def pick_low_level_action(self, maze_state: MazeStateType):
        """ Picks the next low-level action.

        :param maze_state: The Maze state for which to pick an action.
        """
        # State is safe and no queued low-level action: do nothing
        if self.observation_conversion.is_safe(maze_state) and self.low_level_step_count == -1:
            action = (None, None)
            return action

        # Optimize the low-level action every step
        self.low_actions = self.low_level_policy.optimize_low_actions(maze_state=maze_state,
                                                                      low_actions=self.low_actions)
        self.low_level_step_count += 1

        # Queue has been emptied after optimization: do nothing
        if len(self.low_actions) == 0:
            # Specify noop action
            action = (None, None)
            self.low_level_step_count = -1

        # Standard case: execute low-level action from queue
        else:
            action = self.pop_low_action()

        # If max. level of low level step count reached, log and reset
        if self.max_low_level_step_count <= self.low_level_step_count:
            self.low_level_step_count = -1

        return action

    def pop_low_action(self):
        """
        Simply pops next low level action from queue.
        """
        # Pick the next low level action.
        sub_id, new_topo = self.low_actions.pop(0)[:2]
        action = (sub_id, new_topo)
        return action

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        """ Step method that controls if a low-level action is performed and stepped or if control is returned to
        the high-level policy outside of the wrapper.
        """
        goal_logits = action

        # convert action to actual bus action
        bus_goal = self.action_conversion.goal_action_to_bus_action(action)
        order = None
        info = {}
        total_reward = 0
        internal_steps = 0
        internal_rewards = []
        internal_sandbox_scores = []

        is_first_iteration = True
        while True:
            maze_state = self.env.get_maze_state()
            is_safe = self.observation_conversion.is_safe(maze_state=maze_state)
            if self.verbose:
                print(f"Agent->act (is_safe: {is_safe})")

            skip = False
            # Auto-reconnect disconnected power lines
            if False in maze_state.line_status:
                if self.verbose:
                    print("  -> Power line is offline!")
                low_action = self.low_level_policy.reconnect_line(maze_state)
                if low_action is not None:
                    if self.verbose:
                        print("    + Reconnecting power line action selected.")
                    obs, reward, done, info = self.env.step(low_action)

                    # Count internal steps
                    internal_steps += 1
                    self.total_steps_since_reset += 1
                    # Fire internal step event
                    self.smdp_events.internal_step()
                    internal_rewards.append(reward)
                    internal_sandbox_scores.append(info['rewards']['L2RPNSandBoxScore'])

                    total_reward += reward
                    skip = True
                    # Check if env done
                    if done:
                        self.update_info_dict(info, internal_rewards, internal_sandbox_scores, internal_steps)
                        return obs, total_reward, done, info

            # ONLY EXIT CONDITION (besides done): UNSAFE STATE AND LOW_LEVEL_STEP_COUNT == -1. Don't exit if it
            # happens in first iteration.
            if not skip and (not is_safe and self.low_level_step_count == -1) and not is_first_iteration:
                # New goal needs to be generated -> exit step function

                # Record this with an event
                self.smdp_events.new_goal_required()

                self.update_info_dict(info, internal_rewards, internal_sandbox_scores, internal_steps)
                # Remark: obs, reward, done, info have to exist since this branch will never be called in the first
                #         iteration
                return obs, total_reward, done, info

            # Do the following also immediately after a reset (flagged by self.after_reset)
            elif not skip and ((not is_safe and self.low_level_step_count == -1) or self.after_reset):
                # Set after_reset flag to False
                if self.after_reset:
                    self.after_reset = False
                # Plan sub-actions to get to current goal
                self.low_actions = self.low_level_policy.compute_low_level_actions_toward_goal(
                    bus_goal=bus_goal, order=order, maze_state=maze_state)

                # no low-level action left towards target topology
                if len(self.low_actions) == 0:
                    low_action = (None, None)  # self.env.action_conversion.action_space()
                    if self.verbose:
                        print(f'Stepping env with noop action ')
                    obs, reward, done, info = self.env.step(low_action)
                    # Fire event to record causing noop
                    self.smdp_events.no_low_level_action_left()

                    # Count internal steps
                    internal_steps += 1
                    self.total_steps_since_reset += 1
                    # Fire internal step event
                    self.smdp_events.internal_step()

                    internal_rewards.append(reward)
                    internal_sandbox_scores.append(info['rewards']['L2RPNSandBoxScore'])
                    total_reward += reward
                    self.update_info_dict(info, internal_rewards, internal_sandbox_scores, internal_steps)
                    return obs, total_reward, done, info

                self.update_goal_member_variables(goal_logits, bus_goal, self.low_actions, order)

            # sample low level action
            if not skip:
                if self.verbose:
                    print(f"  + Returning low level policy action. [remaining: {len(self.low_actions)}]")
                low_action = self.pick_low_level_action(maze_state=maze_state)

            # Perform low-level action on env
            if self.verbose:
                print(f'Stepping env with action '
                      f'{self.env.action_conversion.space_to_maze(action=low_action, maze_state=maze_state)}')

            # Record what type of action is performed.
            if action == (None, None):
                self.smdp_events.regular_noop()
            else:
                self.smdp_events.regular_action()

            # Step env.
            obs, reward, done, info = self.env.step(low_action)

            # Count internal steps
            internal_steps += 1
            self.total_steps_since_reset += 1
            # Fire internal step event
            self.smdp_events.internal_step()

            internal_rewards.append(reward)
            internal_sandbox_scores.append(info['rewards']['L2RPNSandBoxScore'])
            total_reward += reward

            # for SMAAC-like evaluation: if total steps since last reset exceed 864, exit
            if self.smaac_evaluation_mode and self.total_steps_since_reset >= 864:
                done = True

            # Check if env done
            if done:
                self.update_info_dict(info, internal_rewards, internal_sandbox_scores, internal_steps)
                return obs, total_reward, done, info

            if is_first_iteration:
                is_first_iteration = False

    def update_info_dict(self, info: Dict, internal_rewards: List[float], internal_sandbox_scores: List[float],
                         internal_steps: int):
        """ Updates info dict with information about current trajectory.

        :param info: the info dict to be updated.
        :param internal_rewards: A list of internal rewards that has been accumulated so far.
        :param internal_sandbox_scores: A list of internal L2RPNSandBoxScores that have been accumulated so far.
        :param internal_steps: The number of internal steps performed so far.
        """
        # Add number of internal steps to info
        info['n_internal_steps'] = internal_steps
        # Add internal rewards to info
        info['internal_rewards'] = internal_rewards
        # Add internal rewards to info
        info['internal_sandbox_scores'] = internal_sandbox_scores
        if self.verbose:
            print(f'step took {len(info["low_level_trajectories"])} sub-steps')
