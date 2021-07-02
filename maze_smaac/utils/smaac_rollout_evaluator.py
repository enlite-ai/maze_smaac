"""Performs evaluation rollouts in the supplied env following https://github.com/KAIST-AILab/SMAAC.

Adopted mainly from https://github.com/KAIST-AILab/SMAAC/blob/53c52f35dfa9224d1adfc5d3e9e67912b7cf0f1b/evaluate.py and
https://github.com/KAIST-AILab/SMAAC/blob/53c52f35dfa9224d1adfc5d3e9e67912b7cf0f1b/train.py#L253
"""

import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.core.wrappers.wrapper import ObservationWrapper, Wrapper
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase

import maze_smaac
from maze_smaac.env.maze_env import L2RPNMazeEnv

DATA_DIR = os.path.join(Path(list(maze_smaac.__path__)[0]), "data")

# (train, validation, test) split of chronics
DATA_SPLIT = {
    'l2rpn_wcci_2020': (
        [17, 240, 494, 737, 990, 1452, 1717, 1942, 2204, 2403, 19, 242, 496, 739, 992, 1454, 1719, 1944, 2206, 2405,
         230, 301, 704, 952, 1008, 1306, 1550, 1751, 2110, 2341, 2443, 2689],
        list(range(2880, 2890)), [18, 241, 495, 738, 991, 1453, 1718, 1943, 2205, 2404])
}

# maximum fast forward steps for respective power grids
MAX_FFW = {
    'l2rpn_wcci_2020': 26
}


def read_ffw_json(path, chronics, power_grid):
    res = {}
    for i in chronics:
        for j in range(MAX_FFW[power_grid]):
            with open(os.path.join(path, f'{i}_{j}.json'), 'r', encoding='utf-8') as f:
                a = json.load(f)
                res[(i, j)] = (a['dn_played'], a['donothing_reward'], a['donothing_nodisc_reward'])
            if i >= 2880:
                break
    return res


class SMAACRolloutEvaluator(RolloutEvaluator):
    """Evaluates a given policy as official SMAAC repository does.

    :param eval_env: Not required since we rollout on single custom env.
    :param: n_episodes: Unused in this evaluator so far.
    :param model_selection: Model selection to notify about the recorded rewards.
    :param deterministic: deterministic or stochastic action sampling (selection).
    :param use_test_set: If True test set chronics will be used for rollouts; else validation set chronics.
    """

    def __init__(self,
                 eval_env: Union[L2RPNMazeEnv, SequentialVectorEnv],
                 model_selection: Optional[ModelSelectionBase],
                 n_episodes: int,
                 deterministic: bool,
                 use_test_set: bool
                 ):
        super().__init__(eval_env=eval_env, n_episodes=n_episodes, model_selection=model_selection,
                         deterministic=deterministic)
        # From official rollout evaluator.
        if isinstance(eval_env, SequentialVectorEnv):
            assert len(eval_env.envs) == 1
            self.eval_env = eval_env.envs[0]
        else:
            self.eval_env = eval_env

        self.model_selection = model_selection
        self.use_test_set = use_test_set

        # Compute validation split for grid
        power_grid = self.eval_env.power_grid
        train_chronics, valid_chronics, test_chronics = DATA_SPLIT[power_grid]
        self.chronics = test_chronics if self.use_test_set else valid_chronics

        # Select chronics.
        env_path = os.path.join(DATA_DIR, power_grid)
        dn_json_path = os.path.join(env_path, 'json')
        if self.use_test_set:
            chronics_to_use = test_chronics
        else:
            chronics_to_use = train_chronics + valid_chronics

        self.dn_ffw = read_ffw_json(dn_json_path, chronics_to_use, power_grid)

        self.ep_infos = {}
        if os.path.exists(dn_json_path):
            for i in list(set(chronics_to_use)):
                with open(os.path.join(dn_json_path, f'{i}.json'), 'r', encoding='utf-8') as f:
                    self.ep_infos[i] = json.load(f)

        self.max_ffw = MAX_FFW[power_grid]

        # Define test env
        # self.test_env = make(env_path, test=True, reward_class=L2RPNSandBoxScore, backend=LightSimBackend(),
        #                     other_rewards={'loss': LossReward})
        # Just use the inner wrapped env as test env
        self.test_env = self.eval_env.wrapped_env
        self.test_env.deactivate_forecast()
        self.test_env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
        self.test_env.parameters.NB_TIMESTEP_RECONNECTION = 12
        self.test_env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 3
        self.test_env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 3
        self.test_env.parameters.HARD_OVERFLOW_THRESHOLD = 200.0
        self.test_env.seed(59)

        # Init observation and action conversion
        # self.observation_conversion = self.eval_env.envs[0].observation_conversion
        # self.action_conversion = self.eval_env.envs[0].action_conversion

    def process_observation(self, env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv],
                            observation: ObservationType) -> ObservationType:
        """Get the observation by recursively passing it thorough the wrapper stack.

        :param env: The current env in the wrapper stack.
        :param observation: The current observation in the wrapper stack.

        :return The observation processed by the env, if the the current (or any sub env) is an observation wrapper.
        """
        if isinstance(env, ObservationWrapper):
            return env.observation(self.process_observation(env.env, observation))
        elif isinstance(env, Wrapper):
            return self.process_observation(env.env, observation)
        else:
            return observation

    @override(RolloutEvaluator)
    def evaluate(self, policy: TorchPolicy) -> None:
        """Evaluate given policy (results are stored in stat logs) and dump the model if the reward improved.

        :param policy: Policy to evaluate
        """
        policy.eval()

        result = {}

        for idx, i in enumerate(self.chronics):
            maze_state = None

            ffw = int(np.argmin([self.dn_ffw[(i, fw)][0] for fw in range(self.max_ffw) if
                                 (i, fw) in self.dn_ffw and self.dn_ffw[(i, fw)][0] >= 10]))

            dn_step = self.dn_ffw[(i, ffw)][0]
            self.test_env.seed(59)
            self.test_env.set_id(i)
            obs = self.eval_env.reset()

            if ffw > 0:
                self.test_env.fast_forward_chronics(ffw * 288 - 3)
                maze_state, *_ = self.test_env.step(self.test_env.action_space())

            # Transform maze_state into obs
            if maze_state:
                obs = self.eval_env.maze_env.observation_conversion.maze_to_space(maze_state=maze_state)
                # Pass obs through Maze wrapper stack
                obs = self.process_observation(env=self.eval_env, observation=obs)

            total_reward = 0
            total_sandboxscore = 0
            alive_frame = 0
            done = False
            result[(i, ffw)] = {}
            while not done:
                maze_action = policy.compute_action(obs, actor_id=None, maze_state=None,
                                                    deterministic=self.deterministic)
                obs, reward, done, info = self.eval_env.step(maze_action)
                total_reward += reward
                total_sandboxscore += sum(info['internal_sandbox_scores'])
                # Count internal steps as alive frames
                alive_frame += info['n_internal_steps']
                # stop at 864 frames for evaluation
                if alive_frame >= 864:
                    done = True

            # The L2RPNSandBoxScore is used to calculate the scores.
            l2rpn_score = float(self.compute_episode_score(i, alive_frame, total_sandboxscore, ffw))
            print(f'[Test Ch{i:4d}({ffw:2d})] {alive_frame:3d}/864 ({dn_step:3d}) Score: {l2rpn_score:9.4f}')

        # Enforce the epoch stats calculation (without calling increment_log_step() -- this is up to the trainer)
        self.eval_env.write_epoch_stats()

        # Notify the model selection if available
        if self.model_selection:
            reward = self.eval_env.get_stats_value(BaseEnvEvents.reward, LogStatsLevel.EPOCH, name="mean")
            self.model_selection.update(reward)

    # following competition evaluation script
    def compute_episode_score(self, chronic_id, agent_step, agent_reward, ffw=None):
        min_losses_ratio = 0.7
        ep_marginal_cost = self.test_env.gen_cost_per_MW.max()
        if ffw is None:
            ep_do_nothing_reward = self.ep_infos[chronic_id]["donothing_reward"]
            ep_do_nothing_nodisc_reward = self.ep_infos[chronic_id]["donothing_nodisc_reward"]
            ep_dn_played = self.ep_infos[chronic_id]['dn_played']
            ep_loads = np.array(self.ep_infos[chronic_id]["sum_loads"])
            ep_losses = np.array(self.ep_infos[chronic_id]["losses"])
        else:
            start_idx = 0 if ffw == 0 else ffw * 288 - 2
            end_idx = start_idx + 864
            ep_dn_played, ep_do_nothing_reward, ep_do_nothing_nodisc_reward = self.dn_ffw[(chronic_id, ffw)]
            ep_loads = np.array(self.ep_infos[chronic_id]["sum_loads"])[start_idx:end_idx]
            ep_losses = np.array(self.ep_infos[chronic_id]["losses"])[start_idx:end_idx]

        # Add cost of non delivered loads for blackout steps
        blackout_loads = ep_loads[agent_step:]
        if len(blackout_loads) > 0:
            blackout_reward = np.sum(blackout_loads) * ep_marginal_cost
            agent_reward += blackout_reward

        # Compute ranges
        worst_reward = np.sum(ep_loads) * ep_marginal_cost
        best_reward = np.sum(ep_losses) * min_losses_ratio
        zero_reward = ep_do_nothing_reward
        zero_blackout = ep_loads[ep_dn_played:]
        zero_reward += np.sum(zero_blackout) * ep_marginal_cost
        nodisc_reward = ep_do_nothing_nodisc_reward

        # Linear interp episode reward to codalab score
        if zero_reward != nodisc_reward:
            # DoNothing agent doesnt complete the scenario
            reward_range = [best_reward, nodisc_reward, zero_reward, worst_reward]
            score_range = [100.0, 80.0, 0.0, -100.0]
        else:
            # DoNothing agent can complete the scenario
            reward_range = [best_reward, zero_reward, worst_reward]
            score_range = [100.0, 0.0, -100.0]

        ep_score = np.interp(agent_reward, reward_range, score_range)
        return ep_score
