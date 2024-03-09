from typing import Optional
from typing import Union, Callable

import torch
import torch.nn as nn

from src.data.components import Batch
from src.models.components.atomistic import AtomisticModel
from src.models.components.connectivity import Connectivity, fully_connected
from src.models.components.noise.model import NormalNoiseModel


class TimeConditionedRegressor(nn.Module):
    def __init__(
            self,
            regressor_net: AtomisticModel,
            noise_model: NormalNoiseModel,
            connectivity_module: Optional[Union[Connectivity, Callable[[Batch], Batch]]] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.regressor_net: AtomisticModel = regressor_net
        self.noise_model: NormalNoiseModel = noise_model
        self.connectivity_module: Connectivity = connectivity_module

    def forward(self, noisy_inputs: Batch, ts: torch.LongTensor) -> dict[str, torch.Tensor]:
        assert ts.shape == (noisy_inputs.num_data, 1)
        ts = ts / self.timesteps  # Normalise to [0, 1]
        ts = torch.repeat_interleave(ts, noisy_inputs.num_nodes, dim=0)
        assert ts.shape[0] == noisy_inputs.node_features.shape[0] and ts.shape[1] == 1
        assert ts.min() >= 0 and ts.max() <= 1

        noisy_inputs.ts = ts
        outputs: dict[str, torch.Tensor] = self.regressor_net(noisy_inputs)

        return outputs

    def forward_diffusion(self, inputs: Batch, ts: torch.Tensor) -> Batch:
        (
            noisy_inputs,
            _,
            _,
        ) = self.noise_model.forward_diffusion(inputs, ts)

        # update connectivity
        noisy_inputs = self.update_connectivity(noisy_inputs)
        return noisy_inputs

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
