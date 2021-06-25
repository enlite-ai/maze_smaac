"""Contains a wrapper for the chronics sampling (interesting cases) adopted from
https://github.com/KAIST-AILab/SMAAC/blob/53c52f35dfa9224d1adfc5d3e9e67912b7cf0f1b/train.py """

import json
import os
import random
from typing import Tuple, Dict, Optional

import numpy as np
import torch
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.wrappers.wrapper import Wrapper

import maze_smaac
from maze_smaac.env.events import SMDPEvents
from maze_smaac.env.maze_env import L2RPNMazeEnv
from maze_smaac.utils.smaac_rollout_evaluator import read_ffw_json, DATA_SPLIT, MAX_FFW


class ChronicsSamplingWrapper(Wrapper[MazeEnv]):
    """Resets environment to interesting (hard) chronics and time steps.

    :param env: The environment to wrap.
    :param seed: Numeric seed.
    :param max_steps: Determines after how many steps the env is set to done
    """

    def __init__(self, env: L2RPNMazeEnv, seed: int, max_steps: Optional[int]):
        super(ChronicsSamplingWrapper, self).__init__(env=env, keep_inner_hooks=True)

        # Seed everything
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.env = env
        self.seed_int = seed
        self.max_steps = max_steps

        self.step_counter = 0
        self.record_idx = None

        data_dir = os.path.join(maze_smaac.__path__._path[0], 'data')
        env_path = os.path.join(data_dir, env.power_grid)
        train_chronics, valid_chronics, _ = DATA_SPLIT[env.power_grid]
        dn_json_path = os.path.join(env_path, 'json')

        # select chronics
        self.dn_ffw = read_ffw_json(dn_json_path, train_chronics + valid_chronics, env.power_grid)

        ep_infos = {}
        if os.path.exists(dn_json_path):
            for i in list(set(train_chronics + valid_chronics)):
                with open(os.path.join(dn_json_path, f'{i}.json'), 'r', encoding='utf-8') as f:
                    ep_infos[i] = json.load(f)

        max_ffw = MAX_FFW[env.power_grid]

        # initialize training chronic sampling weights
        self.train_chronics_ffw = [(cid, fw) for cid in train_chronics for fw in range(max_ffw)]
        total_chronic_num = len(self.train_chronics_ffw)
        self.chronic_records = [0] * total_chronic_num
        self.chronic_step_records = [0] * total_chronic_num

        for i in self.chronic_records:
            cid, fw = self.train_chronics_ffw[i]
            self.chronic_records[i] = self.chronic_priority(cid, fw, 1)

        # create event topics
        self.smaac_events = self.core_env.context.event_service.create_event_topic(SMDPEvents)

    def reset(self) -> ObservationType:
        """Reset environment.
        """
        self.step_counter = 0
        # sample training chronic
        dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(self.chronic_records))
        self.record_idx = dist.sample().item()
        self.chronic_id, self.ffw = self.train_chronics_ffw[self.record_idx]
        self.env.wrapped_env.set_id(self.chronic_id)
        self.env.wrapped_env.seed(self.seed_int)
        obs = self.env.reset()
        if self.ffw > 0:
            self.env.wrapped_env.fast_forward_chronics(self.ffw * 288 - 3)
            # step noop action
            obs, *_ = self.env.step((None, None))
            self.smaac_events.noop_in_reset()

        return obs

    def chronic_priority(self, cid: int, ffw: int, step: int) -> float:
        """Compute (priority) weight for chronics sampling.

        :param cid: The chronics id.
        :param ffw: Number of fast forward steps.
        :param step: current step.
        :return: The sampling priority.
        """
        m = 864
        scale = 2.
        diff_coef = 0.05
        d = self.dn_ffw[(cid, ffw)][0]
        progress = 1 - np.sqrt(step / m)
        difficulty = 1 - np.sqrt(d / m)
        score = (progress + diff_coef * difficulty) * scale
        return score

    def step(self, action) -> Tuple[ObservationType, float, bool, Dict]:
        """Take env step.
        """
        obs, reward, done, info = self.env.step(action)
        self.step_counter += 1

        # Set env done if 864 steps have been taken.
        if self.max_steps and self.step_counter == self.max_steps:
            done = True

        if done:
            # update chronic sampling weight
            self.chronic_records[self.record_idx] = self.chronic_priority(self.chronic_id, self.ffw, self.step_counter)
            self.chronic_step_records[self.record_idx] = self.step_counter

        return obs, reward, done, info
