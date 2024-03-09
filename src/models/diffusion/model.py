from typing import Tuple, Callable, Union, Optional, Container

import torch
import torch.nn as nn

from src.data.components import Batch
from src.data.components.datasets.base import BaseDataset
from src.models.atomistic import AtomisticModel
from src.models.connectivity import Connectivity, fully_connected
from src.models.diffusion.noise.model import MaskedNormalNoiseModel


def filter_dict(d: dict, allowed_keys: Container):
    return {key: value for (key, value) in d.items() if key in allowed_keys}


METAL_CENTER_NUMBERS = {
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
}


class ConditionalEmpiricalCountDistribution(nn.Module):
    """
    Two-step sampling procedure:
    1) sample from p(z);
    2) sample from p(x|z).
    """

    def __init__(
            self,
            z_labels: torch.Tensor,
            z_counts: torch.Tensor,
            x_labels: torch.Tensor,
            x_given_z_counts: torch.Tensor,
    ):
        super().__init__()
        assert z_labels.ndim == 1 and z_labels.shape == z_counts.shape
        assert torch.all(z_labels >= 0) and torch.all(z_counts >= 0)
        assert torch.unique(z_labels).shape[0] == z_labels.shape[0]
        assert z_labels.shape[0] == x_given_z_counts.shape[0]

        self.register_buffer("z_labels", z_labels)
        self.register_buffer("z_counts", z_counts)
        self.register_buffer("z_probs", z_counts / z_counts.sum())
        # Create map from labels to index with default value -1
        z_label_index = -torch.ones(self.z_labels.max() + 1, dtype=torch.long)
        z_label_index[self.z_labels] = torch.arange(self.z_labels.shape[0], dtype=torch.long)
        self.register_buffer("z_label_index", z_label_index)

        self.register_buffer("x_labels", x_labels)
        self.register_buffer("x_given_z_counts", z_counts)
        self.register_buffer(
            "x_given_z_probs", x_given_z_counts / x_given_z_counts.sum(dim=1, keepdim=True)
        )
        x_label_index = -torch.ones(self.x_labels.max() + 1, dtype=torch.long)
        x_label_index[self.x_labels] = torch.arange(self.x_labels.shape[0], dtype=torch.long)
        self.register_buffer("x_label_index", x_label_index)

    def sample(
            self, num_samples: int = 1, return_indices: bool = False
    ) -> Union[
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Sample from the distribution."""
        z_indices = torch.multinomial(self.z_probs, num_samples=num_samples, replacement=True)

        return self.sample_conditioned(z_indices, return_indices=return_indices)

    def sample_conditioned(
            self, z_indices: torch.Tensor, return_indices: bool = False
    ) -> Union[
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        x_indices = torch.multinomial(
            self.x_given_z_probs[z_indices], num_samples=1, replacement=True
        ).squeeze()

        z_labels: torch.Tensor = self.z_labels[z_indices]
        x_labels: torch.Tensor = self.x_labels[x_indices]

        if return_indices:
            return (z_indices, z_labels), (x_indices, x_labels)

        return z_labels, x_labels

    def log_prob(self, z_labels: torch.Tensor, x_labels: torch.Tensor) -> torch.Tensor:
        z_indices, x_indices = self.z_label_index[z_labels], self.x_label_index[x_labels]
        zx_indices = [z_indices, x_indices.squeeze()]
        z_probs, x_given_z_probs = (
            self.z_probs[z_indices],
            self.x_given_z_probs[zx_indices],
        )
        return torch.log(z_probs) + torch.log(x_given_z_probs)

    @classmethod
    def from_counter(cls, z_counter: dict[int, int], x_given_z_counter: dict[int, dict[int, int]]):
        z_labels = torch.tensor(list(z_counter.keys()), dtype=torch.long)
        z_counts = torch.tensor(list(z_counter.values()), dtype=torch.long)

        x_labels = torch.tensor(
            list(set.union(*[set(x_given_z_counter[z].keys()) for z in x_given_z_counter])),
            dtype=torch.long,
        )
        x_labels = torch.arange(
            max(x_labels) + 1,
            dtype=torch.long,
        )
        x_given_z_counts = torch.zeros((len(z_labels), max(x_labels) + 1))
        for zi, z in enumerate(z_labels):
            for x in x_given_z_counter[z.item()]:
                x_given_z_counts[zi, x] = x_given_z_counter[z.item()][x]

        return ConditionalEmpiricalCountDistribution(
            z_labels, z_counts, x_labels=x_labels, x_given_z_counts=x_given_z_counts
        )


class OMDiff(nn.Module):
    key_features = "node_features"
    key_positions = "node_positions"

    def __init__(
            self,
            denoising_net: AtomisticModel,
            noise_model: MaskedNormalNoiseModel,
            num_nodes_distribution: ConditionalEmpiricalCountDistribution,
            node_labels: list[int],
            masked_node_labels: list[int],
            connectivity_module: Optional[Union[Connectivity, Callable[[Batch], Batch]]] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.denoising_net: AtomisticModel = denoising_net
        self.noise_model: MaskedNormalNoiseModel = noise_model
        self.num_nodes_distribution: ConditionalEmpiricalCountDistribution = num_nodes_distribution
        self.connectivity_module: Connectivity = connectivity_module

        sorted_node_labels = list(sorted(node_labels))
        node_features_mask = torch.zeros(1, len(node_labels), dtype=torch.bool)
        for masked_label in sorted(masked_node_labels):
            idx = sorted_node_labels.index(masked_label)
            node_features_mask[0, idx] = True

        self.register_buffer("node_features_mask", node_features_mask)

    def forward(self, noisy_inputs: Batch, ts: torch.LongTensor) -> dict[str, torch.Tensor]:
        """
        Forward pass (the backward diffusion process).

        Args:
            noisy_inputs (batch object): Noisy inputs.
            ts (torch.LongTensor): Integer timesteps for the diffusion process with shape (num_data,).
        Returns:
            dict with:
             - predicted positions' noise
             - predicted (clean) node features
        """
        # Prepare node states
        # Concatenate normalised timestep to the node features
        assert ts.shape == (noisy_inputs.num_data, 1)
        ts = ts / self.timesteps  # Normalise to [0, 1]
        ts = torch.repeat_interleave(ts, noisy_inputs.num_nodes, dim=0)
        assert ts.shape[0] == noisy_inputs.node_features.shape[0] and ts.shape[1] == 1
        assert ts.min() >= 0 and ts.max() <= 1

        node_mask = noisy_inputs.node_mask

        initial_node_positions = noisy_inputs.node_positions
        noisy_inputs.ts = ts

        outputs: dict[str, torch.Tensor] = self.denoising_net(noisy_inputs)

        # Compute positions difference
        features_mask = torch.bitwise_or(node_mask, self.node_features_mask)
        node_features = torch.where(
            features_mask,
            torch.zeros_like(outputs["node_states"]),
            outputs["node_states"],
        )

        noise_node_positions = outputs["node_positions"] - initial_node_positions
        noise_node_positions = torch.where(
            node_mask, torch.zeros_like(noise_node_positions), noise_node_positions
        )

        return_dict = {
            self.key_features: node_features,
            self.key_positions: noise_node_positions,
        }
        return return_dict

    def forward_diffusion(
            self, inputs: Batch, ts: torch.Tensor
    ) -> Tuple[Batch, dict[str, torch.Tensor], torch.Tensor]:
        inputs.node_features_mask = self.node_features_mask
        (
            noisy_inputs,
            targets,
            gamma_t,
        ) = self.noise_model.forward_diffusion(inputs, ts)

        # update connectivity
        noisy_inputs = self.update_connectivity(noisy_inputs)

        return (
            noisy_inputs,
            targets,
            gamma_t,
        )

    def update_connectivity(self, inputs: Batch, fully_connected_if_None: bool = False) -> Batch:
        if self.connectivity_module is None:
            if fully_connected_if_None:
                inputs.edge_index, inputs.num_edges = fully_connected(inputs.num_nodes)
            return inputs
        else:
            return self.connectivity_module(inputs)

    @property
    def timesteps(self) -> int:
        return self.noise_model.noise_schedule.timesteps

    @classmethod
    def from_data(
            cls,
            denoising_net: AtomisticModel,
            noise_model: MaskedNormalNoiseModel,
            dataset: BaseDataset,
            connectivity_module: Optional[Union[Connectivity, Callable[[Batch], Batch]]] = None,
            masked_node_labels: Optional[set[int]] = None,
    ):
        num_nodes_distribution, node_labels, masked_node_labels = cls.get_num_nodes_distribution(
            dataset=dataset, masked_node_labels=masked_node_labels
        )

        return cls(
            denoising_net=denoising_net,
            noise_model=noise_model,
            num_nodes_distribution=num_nodes_distribution,
            connectivity_module=connectivity_module,
            node_labels=node_labels,
            masked_node_labels=masked_node_labels,
        )

    @staticmethod
    def get_num_nodes_distribution(
            dataset: BaseDataset,
            masked_node_labels: Optional[set[int]] = None,
    ):
        node_labels = dataset.node_labels

        if masked_node_labels is None:
            masked_node_labels = [n for n in METAL_CENTER_NUMBERS if n in node_labels]

        z_counter, x_given_z_counter = dataset.conditional_node_count
        # filter and keep only metal centers
        z_counter = filter_dict(z_counter, allowed_keys=masked_node_labels)
        x_given_z_counter = filter_dict(x_given_z_counter, allowed_keys=masked_node_labels)

        num_nodes_distribution = ConditionalEmpiricalCountDistribution.from_counter(
            z_counter=z_counter, x_given_z_counter=x_given_z_counter
        )

        return num_nodes_distribution, node_labels, masked_node_labels
