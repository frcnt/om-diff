from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter.composite import scatter_softmax

import src.models.ops as ops
from src.data.components import Batch


class MultiHeadAttentionReadout(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_layers: int,
            num_heads: int,
            readout_net: nn.Module,
            input_key: str,
            splits_key: Optional[str] = None,
            output_key: Optional[str] = None,
    ):
        super().__init__()
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(input_dim, num_heads) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.readout_net = readout_net
        self.input_key = input_key
        self.output_key = input_key if output_key is None else output_key
        self.splits_key = splits_key
        self.model_outputs = [self.output_key]

    def forward(self, inputs: Batch) -> Batch:
        x = getattr(inputs, self.input_key)
        splits = getattr(inputs, self.splits_key)

        x = self.layer_norm(x)
        sequences = torch.split(x, splits.detach().cpu().tolist())
        masks = [
            torch.zeros(size=s.shape[0:1], dtype=torch.bool, device=x.device) for s in sequences
        ]
        masks = nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=True)

        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=False)
        query = torch.ones(
            size=(1, padded.shape[1], padded.shape[2]),
            device=padded.device,
            dtype=padded.dtype,
        )
        for att_layer in self.attention_layers:
            query, _ = att_layer(query, padded, padded, key_padding_mask=masks)

        x = self.readout_net(query.squeeze(0))

        setattr(inputs, self.output_key, x)
        return inputs


class GatedEquivariantBlock(nn.Module):
    def __init__(
            self,
            hidden_channels,
            out_channels,
            intermediate_channels=None,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            nn.SiLU(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(
            self,
            atom_states_scalar: torch.Tensor,
            atom_states_vector: torch.Tensor,
    ):
        vec1 = torch.norm(self.vec1_proj(atom_states_vector), dim=-2)
        vec2 = self.vec2_proj(atom_states_vector)

        x = torch.cat([atom_states_scalar, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = F.silu(x)

        return x, v


class EquivariantReadout(nn.Module):
    def __init__(
            self,
            hidden_channels,
    ):
        super(EquivariantReadout, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(
            self,
            atom_states_scalar: torch.Tensor,
            atom_states_vector: torch.Tensor,
    ):
        for layer in self.output_network:
            atom_states_scalar, atom_states_vector = layer(atom_states_scalar, atom_states_vector)
        return atom_states_scalar, atom_states_vector.squeeze()


class Readout(nn.Module):
    def __init__(
            self,
            net: nn.Module,
            input_key: str,
            output_key: Optional[str] = None,
            reduction: Optional[str] = None,
            splits_key: Optional[str] = None,
    ):
        super().__init__()

        self.net = net
        self.input_key = input_key
        self.output_key = input_key if output_key is None else output_key
        self.reduction = reduction
        if reduction:
            assert (
                    splits_key is not None
            ), "Argument 'splits_key' should be provided when using 'reduction'."
        self.splits_key = splits_key
        self.model_outputs = [self.output_key]

    def forward(self, inputs: Batch) -> Batch:
        x = getattr(inputs, self.input_key)
        x = self.net(x)
        if self.reduction:
            splits = getattr(inputs, self.splits_key)
            x = ops.reduce_splits(x, splits, reduction=self.reduction)
        setattr(inputs, self.output_key, x)
        return inputs


class Set2VecReadout(nn.Module):
    """
    Converts a set to a vector
    """

    def __init__(
            self,
            score_net: nn.Module,
            readout_net: nn.Module,
            input_key: str,
            splits_key: Optional[str] = None,
            output_key: Optional[str] = None,
    ):
        super().__init__()
        self.score_net = score_net
        self.readout_net = readout_net
        self.input_key = input_key
        self.output_key = input_key if output_key is None else output_key
        self.splits_key = splits_key
        self.model_outputs = [self.output_key]

    def forward(self, inputs: Batch) -> Batch:
        x = getattr(inputs, self.input_key)
        splits = getattr(inputs, self.splits_key)

        s = self.score_net(x)
        s = scatter_softmax(s, index=inputs.index[:, None], dim=0)
        sx = ops.sum_splits(s * x, splits)
        out = self.readout_net(sx)

        setattr(inputs, self.output_key, out)
        return inputs
