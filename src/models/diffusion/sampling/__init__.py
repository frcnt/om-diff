import copy
from typing import Tuple, Iterator, Optional

import torch

from src.data.components import Batch, AtomsData, collate_data
from src.models.diffusion.model import OMDiff


class BaseSampler:
    def compute_noise(
            self, om_diff: OMDiff, inputs: Batch, ts: torch.LongTensor
    ) -> dict[str, torch.Tensor]:
        return om_diff.forward(inputs, ts)

    @torch.no_grad()
    def sample_prior(
            self,
            om_diff: OMDiff,
            num_nodes: torch.Tensor,
            condition_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> Batch:
        """Sample a batch of data from the prior."""
        data_list = []
        if condition_dict is None:
            condition_dict = {}
        for n in num_nodes:
            n = n.item()
            # Sample data object with n nodes
            prior_features = om_diff.noise_model.sample_prior_one(n, device=torch.device("cpu"))
            data = AtomsData(
                **prior_features,
                **condition_dict,
            )
            data_list.append(data)
        batch = collate_data(data_list)
        batch = om_diff.update_connectivity(inputs=batch, fully_connected_if_None=True)

        batch = batch.to(num_nodes.device)

        assert batch.node_positions.sum() < 5e-2  # Is centered
        return batch

    @torch.no_grad()
    def sample_intermediate_step(self, om_diff: OMDiff, inputs: Batch, t: torch.LongTensor) -> Batch:
        assert torch.numel(t) == 1
        # Predict noise with the model
        ts = torch.full(
            (inputs.num_data, 1),
            fill_value=t.item(),
            device=inputs.node_features.device,
        )

        denoised_inputs = copy.deepcopy(inputs)

        predictions = self.compute_noise(om_diff=om_diff, inputs=inputs, ts=ts)

        denoised_inputs = om_diff.noise_model.backward_diffusion(denoised_inputs, predictions, t)

        denoised_inputs = om_diff.update_connectivity(denoised_inputs, fully_connected_if_None=False)

        return denoised_inputs

    @torch.no_grad()
    def sample_final_step(self, om_diff: OMDiff, inputs: Batch) -> Batch:
        """Sample the backward diffusion process at the final timestep t=0."""

        device = next(om_diff.parameters()).device
        denoised_inputs = copy.deepcopy(inputs)

        ts = torch.zeros((denoised_inputs.num_data, 1), dtype=torch.long, device=device)
        predictions = self.compute_noise(om_diff, inputs, ts)

        denoised_inputs = om_diff.noise_model.sample_final_step(denoised_inputs, predictions)
        denoised_inputs = om_diff.update_connectivity(denoised_inputs, fully_connected_if_None=False)

        return denoised_inputs

    @torch.no_grad()
    def generate(
            self,
            om_diff: OMDiff,
            condition_dict: Optional[dict[str, torch.Tensor]] = None,
            num_samples: int = 1,
            num_nodes: torch.Tensor = None,
            yield_init: bool = False,
            yield_every: int = 1,
    ) -> Iterator[Tuple[int, Batch]]:  # return type generator
        device = next(om_diff.parameters()).device
        if num_nodes is None:
            num_nodes = om_diff.num_nodes_distribution.sample(num_samples)

        yield_every = yield_every if yield_every > 0 else om_diff.timesteps
        # Sample t=T from prior
        samples = self.sample_prior(edm=om_diff, num_nodes=num_nodes, condition_dict=condition_dict)

        if yield_init:
            yield om_diff.timesteps, samples
        # Sample t-1 given t for t=T,...,1
        for t in torch.arange(om_diff.timesteps, 1, -1, device=device, dtype=torch.long):
            t = t.view(-1, 1)
            samples = self.sample_intermediate_step(om_diff=om_diff, inputs=samples, t=t)
            if (t - 1) > 0 and ((t - 1) % yield_every) == 0:
                yield t - 1, samples
        # Sample output given t=0
        samples = self.sample_final_step(om_diff=om_diff, inputs=samples)
        yield 0, samples
