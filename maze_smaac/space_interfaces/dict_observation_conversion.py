"""Contains a Maze ObservationConversion interface. Contains code adopted from
https://github.com/KAIST-AILab/SMAAC/blob/53c52f35dfa9224d1adfc5d3e9e67912b7cf0f1b/converter.py """
from typing import Tuple

import grid2op
import numpy as np
import torch
from grid2op.Observation import CompleteObservation
from gym import spaces
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationConversionInterface, ObservationType


class MazeSMAACObservationConversion(ObservationConversionInterface):
    """ L2RPN MazeState to dictionary ObservationConversion

    :param env: A grid2op Environment.
    :param mask: Lower substation mask parameter.
    :param mask_hi: Upper substation mask parameter.
    :param danger: The danger threshold above which to act. delta_t in the paper.
    :param dim_topo: Topology dimension
    :param output_dim: Actor output dimension
    :param n_history: The number of observations to stack.
    """
    def __init__(self, env: grid2op.Environment.Environment, mask: int, mask_hi: int, danger: float,
                 dim_topo: int, output_dim: int, n_history: int):
        self.mask = mask
        self.mask_hi = mask_hi
        self.dim_topo = dim_topo
        self.output_dim = output_dim
        self.danger = danger
        self.n_history = n_history

        self.thermal_limit = env._thermal_limit_a
        self.thermal_limit_under400 = env._thermal_limit_a < 400
        self.obs_space = env.observation_space
        self.action_space = env.action_space

        self._init_obs_converter()

        self.n_features = 5
        self.stacked_obs = []

    def space(self) -> spaces.Dict:
        """ Returns gym observation space """
        float_max = np.finfo(np.float32).max
        float_min = np.finfo(np.float32).min

        input_dim = self.n_features * self.n_history

        space = spaces.Dict({
            'independent_of_action': spaces.Box(low=float_min, high=float_max, shape=[self.dim_topo, input_dim],
                                                dtype=np.float32),
            'dependent_on_action': spaces.Box(low=0, high=1, shape=[self.dim_topo, self.dim_topo], dtype=np.int),
            'goal_topology': spaces.Box(low=-np.float(1), high=np.float(1), shape=(self.output_dim,)),
            'topo': spaces.Box(low=-np.float(1), high=np.float(1), shape=(self.dim_topo, 1)),
        })
        return space

    def maze_to_space(self, maze_state: CompleteObservation) -> ObservationType:
        """Convert environment (simulation) state to agent observation.

        :param maze_state: The environment observation (here a grid2op CompleteObservation).
        :return: the agent observation.
        """

        # get vectorized state
        obs_vect = maze_state.to_vect()[np.newaxis, :]
        obs_vect, topo = self._convert_obs(obs_vect)

        # stack observations for temporal context
        if len(self.stacked_obs) == 0:
            for _ in range(self.n_history):
                self.stacked_obs.append(obs_vect)
        else:
            self.stacked_obs.pop(0)
            self.stacked_obs.append(obs_vect)

        stacked_obs = np.concatenate(self.stacked_obs, axis=-1)
        stacked_obs = np.squeeze(stacked_obs.astype(np.float32), axis=0)

        # compute adjacency matrix
        adj = (maze_state.connectivity_matrix() + np.eye(maze_state.dim_topo, dtype=np.float32))

        # prepare current topology vector for model
        topo = np.squeeze(topo.astype(np.float32), axis=0)

        return {'independent_of_action': stacked_obs, 'dependent_on_action': adj, 'topo': topo}

    def is_safe(self, maze_state: MazeStateType) -> bool:
        """Checks weather the current state is save or requires intervention by the agent.

        :param maze_state: A MazeState
        :return: True if state is safe, else False.
        """

        # iterate power lines
        for ratio, limit in zip(maze_state.rho, self.thermal_limit):
            # distinguish big line and small line
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False

        # all good!
        return True

    def _init_obs_converter(self) -> None:
        """Initialize observation conversion"""
        self.idx = self.obs_space.shape
        self.pp = np.arange(sum(self.idx[:6]), sum(self.idx[:7]))
        self.lp = np.arange(sum(self.idx[:9]), sum(self.idx[:10]))
        self.op = np.arange(sum(self.idx[:12]), sum(self.idx[:13]))
        self.ep = np.arange(sum(self.idx[:16]), sum(self.idx[:17]))
        self.rho = np.arange(sum(self.idx[:20]), sum(self.idx[:21]))
        self.topo = np.arange(sum(self.idx[:23]), sum(self.idx[:24]))
        self.main = np.arange(sum(self.idx[:26]), sum(self.idx[:27]))
        self.over = np.arange(sum(self.idx[:22]), sum(self.idx[:23]))

        # parse substation info
        self.subs = [{'e': [], 'o': [], 'g': [], 'l': []} for _ in range(self.action_space.n_sub)]
        for gen_id, sub_id in enumerate(self.action_space.gen_to_subid):
            self.subs[sub_id]['g'].append(gen_id)
        for load_id, sub_id in enumerate(self.action_space.load_to_subid):
            self.subs[sub_id]['l'].append(load_id)
        for or_id, sub_id in enumerate(self.action_space.line_or_to_subid):
            self.subs[sub_id]['o'].append(or_id)
        for ex_id, sub_id in enumerate(self.action_space.line_ex_to_subid):
            self.subs[sub_id]['e'].append(ex_id)

        self.sub_to_topos = []  # [0]: [0, 1, 2], [1]: [3, 4, 5, 6, 7, 8]
        for sub_info in self.subs:
            a = []
            for i in sub_info['e']:
                a.append(self.action_space.line_ex_pos_topo_vect[i])
            for i in sub_info['o']:
                a.append(self.action_space.line_or_pos_topo_vect[i])
            for i in sub_info['g']:
                a.append(self.action_space.gen_pos_topo_vect[i])
            for i in sub_info['l']:
                a.append(self.action_space.load_pos_topo_vect[i])
            self.sub_to_topos.append(torch.LongTensor(a))

        # split topology over sub_id
        self.sub_to_topo_begin, self.sub_to_topo_end = [], []
        idx = 0
        for num_topo in self.action_space.sub_info:
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)
        self.max_n_line = max([len(topo['o'] + topo['e']) for topo in self.subs])
        self.max_n_or = max([len(topo['o']) for topo in self.subs])
        self.max_n_ex = max([len(topo['e']) for topo in self.subs])
        self.max_n_g = max([len(topo['g']) for topo in self.subs])
        self.max_n_l = max([len(topo['l']) for topo in self.subs])
        self.n_feature = 6

    def _convert_obs(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert and filter observation. After this function, the observation has been reduced to use
        the 5 features active power, rho, bool if electricity larger than threshold, topology configuration,
        time step overflow, maintenance, hazard.
        """
        length = self.action_space.dim_topo  # N

        # active power
        p_ = np.zeros(shape=(obs.shape[0], length))  # (B, N)
        p_[..., self.action_space.gen_pos_topo_vect] = obs[..., self.pp]
        p_[..., self.action_space.load_pos_topo_vect] = obs[..., self.lp]
        p_[..., self.action_space.line_or_pos_topo_vect] = obs[..., self.op]
        p_[..., self.action_space.line_ex_pos_topo_vect] = obs[..., self.ep]

        # capacity of power lines
        rho_ = np.zeros(shape=(obs.shape[0], length))
        rho_[..., self.action_space.line_or_pos_topo_vect] = obs[..., self.rho]
        rho_[..., self.action_space.line_ex_pos_topo_vect] = obs[..., self.rho]

        # danger: (true if electricity flow of a line larger than predefined threshold)
        danger_ = np.zeros(shape=(obs.shape[0], length))
        danger = ((obs[..., self.rho] >= self.danger - 0.05) & self.thermal_limit_under400) | (
                obs[..., self.rho] >= self.danger)
        danger_[..., self.action_space.line_or_pos_topo_vect] = danger.astype(np.float32)
        danger_[..., self.action_space.line_ex_pos_topo_vect] = danger.astype(np.float32)

        # time step overflow
        over_ = np.zeros(shape=(obs.shape[0], length))
        over_[..., self.action_space.line_or_pos_topo_vect] = obs[..., self.over] / 3
        over_[..., self.action_space.line_ex_pos_topo_vect] = obs[..., self.over] / 3

        # line in maintenance?
        main_ = np.zeros(shape=(obs.shape[0], length))
        temp = np.zeros_like(obs[..., self.main])
        temp[obs[..., self.main] == 0] = 1
        main_[..., self.action_space.line_or_pos_topo_vect] = temp
        main_[..., self.action_space.line_ex_pos_topo_vect] = temp

        topo_ = obs[..., self.topo] - 1
        converted_obs = np.stack([p_, rho_, danger_, over_, main_], axis=2)  # B, N, F

        return converted_obs, topo_[:, :, np.newaxis]
