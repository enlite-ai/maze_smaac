"""Implementation of the SMAAC Networks using Maze-Perception blocks"""

from typing import Dict, Sequence

import torch.nn as nn
import torch.nn.functional as F
from maze.perception.blocks import PerceptionBlock
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.general.functional import FunctionalBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc

from maze_smaac.models.maze_smaac_gat_block import SMAACGATBlock


class MazeSMAACBaseNet(nn.Module):
    """Adoption of the SMAAC embedding network

    :param obs_shapes: The shapes of all observations as a dict.
    :param dropout: Dropout parameter for models.
    :param state_dim: Embedding dimension.
    :param nheads: Number of heads for the MultiHeadAttention nets.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], dropout: int, state_dim: int, nheads: int):
        nn.Module.__init__(self)

        self.obs_shapes = obs_shapes
        self.perception_dict: Dict[str, PerceptionBlock] = dict()

        self.perception_dict['linear_layer'] = DenseBlock(
            in_keys=['independent_of_action'],
            in_shapes=obs_shapes['independent_of_action'],
            out_keys='linear_layer',
            hidden_units=[state_dim],
            non_lin=nn.Identity
        )

        self.perception_dict[f'latent_attention'] = SMAACGATBlock(
            in_keys=['linear_layer', 'dependent_on_action'],
            in_shapes=self.perception_dict['linear_layer'].out_shapes() + [obs_shapes['dependent_on_action']],
            out_keys=f'latent_attention', nheads=nheads, dropout=dropout, number_of_gat_layers=6
        )

        self.used_in_keys = ['independent_of_action', 'dependent_on_action']


class MazeSMAACPolicyNet(MazeSMAACBaseNet):
    """Adoption of the SMAAC policy network

    :param obs_shapes: Dictionary mapping of observation names to shapes.
    :param action_logits_shapes: Dictionary mapping of observation names to shapes.
    :param dropout: Dropout parameter for models.
    :param state_dim: Embedding dimension.
    :param nheads: Number of heads for the MultiHeadAttention nets.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 action_logits_shapes: Dict[str, Sequence[int]],
                 dropout: int, state_dim: int, nheads: int):
        super().__init__(obs_shapes, dropout, state_dim, nheads)

        self.perception_dict[f'head_attention'] = SMAACGATBlock(
            in_keys=['latent_attention', 'dependent_on_action'],
            in_shapes=self.perception_dict['latent_attention'].out_shapes() + [obs_shapes['dependent_on_action']],
            out_keys=f'head_attention', nheads=nheads, dropout=dropout, number_of_gat_layers=3)

        self.perception_dict['down_sample'] = DenseBlock(
            in_keys='head_attention', in_shapes=self.perception_dict['head_attention'].out_shapes(),
            out_keys='down_sample',
            hidden_units=[1],
            non_lin=nn.Identity
        )
        self.perception_dict['squeeze'] = FunctionalBlock(
            in_keys='down_sample', in_shapes=self.perception_dict['down_sample'].out_shapes(),
            out_keys='squeeze',
func=lambda x: x.squeeze(-1))
        self.perception_dict['squeeze_topo'] = FunctionalBlock(
            in_keys='topo', in_shapes=obs_shapes['topo'], out_keys='squeeze_topo',
func=lambda x: x.squeeze(-1))
        self.perception_dict['concat'] = ConcatenationBlock(
            in_keys=['squeeze', 'squeeze_topo'], in_shapes=self.perception_dict['squeeze'].out_shapes() +
                                                                  self.perception_dict['squeeze_topo'].out_shapes(),
            out_keys='concat', concat_dim=-1
        )
        self.perception_dict['leaky_relu'] = FunctionalBlock(
            in_keys='concat', in_shapes=self.perception_dict['concat'].out_shapes(),
            out_keys='leaky_relu',
func=lambda x: F.leaky_relu(x))

        # initialize model weights
        module_init = make_module_init_normc(std=1.0)
        for key in self.perception_dict.keys():
            self.perception_dict[key].apply(module_init)

        # build action head
        for action, shape in action_logits_shapes.items():
            self.perception_dict[action] = LinearOutputBlock(in_keys="leaky_relu", out_keys=action,
                                                             in_shapes=self.perception_dict["leaky_relu"].out_shapes(),
                                                             output_units=action_logits_shapes[action][-1])

            module_init = make_module_init_normc(std=0.01)
            self.perception_dict[action].apply(module_init)

        # compile inference model
        self.used_in_keys.append('topo')
        self.net = InferenceBlock(in_keys=self.used_in_keys,
                                  out_keys=list(action_logits_shapes.keys()),
                                  in_shapes=[obs_shapes[key] for key in self.used_in_keys],
                                  perception_blocks=self.perception_dict)

    def forward(self, x):
        """ forward pass. """
        return self.net(x)


class MazeSMAACStateActionValueNet(MazeSMAACBaseNet):
    """Adoption of the SMAAC (state-action) value network

    :param obs_shapes: Dictionary mapping of observation names to shapes.
    :param output_shapes: Dictionary mapping of observation names to shapes.
    :param dropout: Dropout parameter for models.
    :param state_dim: Embedding dimension.
    :param nheads: Number of heads for the MultiHeadAttention nets.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 output_shapes: Dict[str, Sequence[int]],
                 dropout: int, state_dim: int, nheads: int):
        super().__init__(obs_shapes, dropout, state_dim, nheads)

        # build perception part
        self.perception_dict[f'head_attention'] = SMAACGATBlock(
            in_keys=['latent_attention', 'dependent_on_action'],
            in_shapes=self.perception_dict['latent_attention'].out_shapes() + [obs_shapes['dependent_on_action']],
            out_keys=f'head_attention', nheads=nheads, dropout=dropout, number_of_gat_layers=1
        )

        self.perception_dict['down_sample'] = DenseBlock(
            in_keys='head_attention', in_shapes=self.perception_dict['head_attention'].out_shapes(),
            out_keys='down_sample',
            hidden_units=[1],
            non_lin=nn.Identity
        )
        self.perception_dict['squeeze'] = FunctionalBlock(
            in_keys='down_sample', in_shapes=self.perception_dict['down_sample'].out_shapes(),
            out_keys='squeeze',
func=lambda x: x.squeeze(-1))
        self.perception_dict['concat'] = ConcatenationBlock(
            in_keys=['squeeze', 'goal_topology'], in_shapes=self.perception_dict['squeeze'].out_shapes() +
                                                            [obs_shapes['goal_topology']],
            out_keys='concat', concat_dim=-1
        )
        hidden_dim = self.perception_dict['concat'].out_shapes()[0][-1] // 4

        self.perception_dict['hidden'] = DenseBlock(
            in_keys='concat', in_shapes=self.perception_dict['concat'].out_shapes(),
            out_keys='hidden',
            hidden_units=[hidden_dim],
            non_lin=nn.Identity
        )

        self.perception_dict['leaky_relu'] = FunctionalBlock(
            in_keys='hidden', in_shapes=self.perception_dict['hidden'].out_shapes(),
            out_keys='leaky_relu',
func=lambda x: F.leaky_relu(x))

        # initialize model weights
        module_init = make_module_init_normc(std=1.0)
        for key in self.perception_dict.keys():
            self.perception_dict[key].apply(module_init)

        module_init = make_module_init_normc(std=0.01)
        for output_key, output_shape in output_shapes.items():
            self.perception_dict[output_key] = LinearOutputBlock(
                in_keys="leaky_relu", out_keys=output_key,
                in_shapes=self.perception_dict["leaky_relu"].out_shapes(),
                output_units=output_shape[-1])

            self.perception_dict[output_key].apply(module_init)

        # compile inference model
        self.used_in_keys.append('goal_topology')
        self.net = InferenceBlock(in_keys=self.used_in_keys,
                                  out_keys=list(output_shapes.keys()),
                                  in_shapes=[obs_shapes[key] for key in self.used_in_keys],
                                  perception_blocks=self.perception_dict)

    def forward(self, x):
        """ forward pass. """
        return self.net(x)
