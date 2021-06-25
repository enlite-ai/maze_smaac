""" Contains an wrapped version of the SMAAC GAT block using Maze-Perception blocks."""
from typing import Union, List, Sequence, Dict

import torch
from maze.core.annotations import override
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock
from torch import nn

from maze_smaac.models.smaac_models import GATLayer


class SMAACGATBlock(ShapeNormalizationBlock):
    """A wrapped version of the SMAAC GAT block using Maze-Perception blocks.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param nheads: The number of mutli-attention heads.
    :param dropout: The dropout to use in the Gat layers.
    :param number_of_gat_layers: The number of GAT layers to use.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 nheads: int, dropout: float, number_of_gat_layers: int):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=[3, 3],
                         out_num_dims=3)
        self.nheads = nheads
        assert len(self.in_keys) == 2, 'Two keys expected, one for features one for the adj'
        self.output_dim = self.in_shapes[0][-1]
        self.dropout = dropout
        self.number_of_gat_layers = number_of_gat_layers

        self.layer_dict = nn.ModuleDict()
        for i in range(number_of_gat_layers):
            self.layer_dict[f'gat_{i}'] = GATLayer(self.output_dim, nheads, dropout)

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """
        # prepare input tensor
        input_tensor = block_input[self.in_keys[0]]
        adj_tensor = block_input[self.in_keys[1]]

        # forward pass
        tmp_out_tensor = input_tensor
        for gat_layer in self.layer_dict.values():
            tmp_out_tensor = gat_layer(tmp_out_tensor, adj_tensor)

        return {self.out_keys[0]: tmp_out_tensor}

    def __repr__(self):
        txt = f'SMAAC GAT Block x {self.number_of_gat_layers}'
        txt += f'\n\t# of attn heads: {self.nheads}'
        txt += f'\n\tinput/output dim: {self.output_dim}'
        txt += f'\n\tdim of each head: {self.output_dim // 4}'
        txt += f'\n\t[ MultiHeadAttention({self.nheads}, {self.output_dim}, {self.output_dim // 4}, ' \
               f'dropout={self.dropout})'
        txt += f'\n\tPositionwiseFeedForward({self.output_dim}, {self.output_dim}, dropout={self.dropout}) ] x ' \
               f'{self.number_of_gat_layers}'
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
