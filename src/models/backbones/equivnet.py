import dataclasses
from typing import Optional

import torch
import torch.nn as nn

import src.models.ops as ops
from src.data.components import Batch
from src.models.layers.features import PairwiseLayer
from src.models.layers.mlp import MLP
from src.models.layers.norm import GraphLayerNorm
from src.models.layers.rbf import RBFLayer, EnvelopLayer
from src.models.layers.readout import EquivariantReadout


def _assert_for_nan(t: torch.Tensor, msg: str = "NaN detected"):
    assert not torch.any(torch.isnan(t)), msg


@dataclasses.dataclass
class EquivNetHParams:
    num_interactions: int
    input_size: int
    node_size: int
    edge_size: int
    with_edge_interactions: bool
    update_node_positions: bool


class InteractionLayer(nn.Module):
    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
    ):
        super(InteractionLayer, self).__init__()
        self.node_dim = node_dim
        self.W = nn.Linear(edge_dim, 3 * node_dim)
        self.msg_nn = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3 * node_dim),
        )
        self.edge_inference_nn = nn.Sequential(
            nn.Linear(node_dim, 1),
            nn.Sigmoid(),
        )

        self.ln_node = GraphLayerNorm(node_dim, affine=True)

    def forward(
            self,
            node_states_s: torch.Tensor,
            node_states_v: torch.Tensor,
            edge_states: torch.Tensor,
            unit_vectors: torch.Tensor,
            edges: torch.Tensor,
            splits: torch.Tensor,
    ):
        node_states_s = self.ln_node(node_states_s, splits)

        W = self.W(edge_states)
        phi = self.msg_nn(node_states_s)
        Wphi = W * phi[edges[:, 0]]  # num_edges, 3*node_size
        phi_s, phi_vv, phi_vs = torch.split(Wphi, self.node_dim, dim=1)
        edge = self.edge_inference_nn(phi_s)
        messages_scalar = phi_s * edge
        messages_vector = (
                                  node_states_v[edges[:, 0]] * phi_vv[:, None, :]
                                  + phi_vs[:, None, :] * unit_vectors[..., None]
                          ) * edge[..., None]
        reduced_messages_scalar = ops.sum_index(
            messages_scalar, edges[:, 1], out=torch.zeros_like(node_states_s)
        )
        reduced_messages_vector = ops.sum_index(
            messages_vector, edges[:, 1], out=torch.zeros_like(node_states_v)
        )

        return (
            node_states_s + reduced_messages_scalar,
            node_states_v + reduced_messages_vector,
        )


class UpdateLayer(nn.Module):
    def __init__(
            self,
            node_dim: int,
    ):
        super(UpdateLayer, self).__init__()
        self.node_dim = node_dim
        self.UV = nn.Linear(node_dim, 2 * node_dim, bias=False)
        self.UV_nn = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3 * node_dim),
        )

    def forward(self, node_states_s: torch.Tensor, node_states_v: torch.Tensor):
        UVv = self.UV(node_states_v)  # (n_nodes, 3, 2 * F)
        Uv, Vv = torch.split(UVv, self.node_dim, -1)  # (n_nodes, 3, F)
        Vv_norm = torch.sqrt(torch.sum(Vv ** 2, dim=1) + 1e-6)  # norm over spatial components

        a = self.UV_nn(torch.cat((Vv_norm, node_states_s), dim=1))
        a_vv, a_sv, a_ss = torch.split(a, self.node_dim, dim=1)

        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_ss + a_sv * inner_prod
        delta_v = a_vv[:, None, :] * Uv  # a_vv.shape = (n_nodes, F)

        return node_states_s + delta_s, node_states_v + delta_v


class EdgeLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, residual: bool = False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, 2 * node_dim),
            nn.SiLU(),
            nn.Linear(2 * node_dim, edge_dim),
        )
        self.residual = residual
        self.mask = nn.Parameter(
            torch.as_tensor([1.0 for _ in range(edge_dim)]), requires_grad=True
        )

    def forward(
            self, node_states: torch.Tensor, edge_states: torch.Tensor, edges: torch.LongTensor
    ):
        concat_states = torch.cat(
            (node_states[edges].view(-1, 2 * self.node_dim), edge_states), axis=1
        )
        if self.residual:
            return self.mask[None, :] * edge_states + self.edge_nn(concat_states)
        else:
            return self.edge_nn(concat_states)


class EquivNet(nn.Module):
    def __init__(
            self,
            hparams: EquivNetHParams,
            rbf_layer: Optional[RBFLayer] = None,
            envelop_layer: Optional[EnvelopLayer] = None,
            **kwargs,
    ):
        super(EquivNet, self).__init__(**kwargs)
        self.hparams = hparams

        self.project_layer = MLP(
            input_dim=hparams.input_size,
            hidden_dim=hparams.input_size,
            output_dim=hparams.node_size,
        )

        self.rbf_layer = rbf_layer
        self.envelop_layer = envelop_layer

        self.pairwise_layer = PairwiseLayer()

        self.edge_featurizers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1 if rbf_layer is None else rbf_layer.n_features, hparams.edge_size),
                    nn.SiLU(),
                )
                for _ in range(hparams.num_interactions)
            ]
        )

        self.interactions = nn.ModuleList(
            [
                InteractionLayer(hparams.node_size, hparams.edge_size)
                for _ in range(hparams.num_interactions)
            ]
        )
        self.updates = nn.ModuleList(
            [UpdateLayer(hparams.node_size) for _ in range(hparams.num_interactions)]
        )

        if hparams.with_edge_interactions:
            self.edge_interactions = nn.ModuleList(
                [
                    EdgeLayer(hparams.node_size, hparams.edge_size)
                    for _ in range(hparams.num_interactions)
                ]
            )
        else:
            self.edge_interactions = [
                lambda node_states_s, edge_states, edges: edge_states
                for _ in range(hparams.num_interactions)
            ]

        if hparams.update_node_positions:
            self.equivariant_readout = EquivariantReadout(hidden_channels=hparams.node_size)

    def forward(self, inputs: Batch) -> Batch:
        node_states_v = inputs.node_positions.new_zeros(
            (*inputs.node_positions.shape, self.hparams.node_size)
        )

        node_states_s = getattr(inputs, "node_states", inputs.node_features)
        node_states_s = self.project_layer(node_states_s)

        edge_distances, unit_vectors = self.pairwise_layer(
            inputs.node_positions, inputs.edge_index
        )
        if self.rbf_layer is None:
            edge_embedding = edge_distances
        else:
            edge_embedding = self.rbf_layer(edge_distances)

        if self.envelop_layer is not None:
            edge_embedding = self.envelop_layer(edge_embedding) * edge_embedding

        for (
                edge_featurizer,
                edge_interaction,
                interaction,
                update,
        ) in zip(self.edge_featurizers, self.edge_interactions, self.interactions, self.updates):
            edge_states = edge_featurizer(edge_embedding)
            edge_states = edge_interaction(node_states_s, edge_states, inputs.edge_index)
            node_states_s, node_states_v = interaction(
                node_states_s,
                node_states_v,
                edge_states,
                unit_vectors,
                inputs.edge_index,
                inputs.num_nodes,
            )
            node_states_s, node_states_v = update(node_states_s, node_states_v)

        if self.hparams.update_node_positions:
            _, delta_node_positions = self.equivariant_readout.forward(
                node_states_s, node_states_v
            )

            inputs.node_positions = inputs.node_positions + delta_node_positions

        inputs.node_states = node_states_s

        return inputs
