import copy
from typing import Tuple, Iterator, Optional, Union, Callable

import torch
import torch.nn.functional as F

from src.data.components import Batch, AtomsData, collate_data
from src.models.diffusion.model import OMDiff
from src.models.diffusion.sampling import BaseSampler


class MaskedSampler(BaseSampler):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def compute_noise(self, om_diff: OMDiff, inputs: Batch, ts: torch.Tensor):
        return om_diff.forward(inputs, ts)

    @torch.no_grad()
    def sample_prior(
            self,
            om_diff: OMDiff,
            zs_idx: torch.Tensor,
            num_nodes: torch.Tensor,
    ) -> Batch:
        device = torch.device("cpu")
        node_features_mask = torch.clone(om_diff.node_features_mask).to(device)

        data_list = []
        masks = dict(node_features=node_features_mask)
        for z_idx, n in zip(zs_idx, num_nodes):
            n = n.item()
            prior_features = om_diff.noise_model.sample_prior_one(
                n - 1, device=torch.device("cpu"), masks=masks
            )

            # create node_features, node_positions and mask for center
            center_pos = torch.zeros(1, prior_features["node_positions"].shape[1], device=device)
            center_features = torch.zeros_like(
                om_diff.node_features_mask, device=device, dtype=torch.float
            )
            idx_true = torch.nonzero(om_diff.node_features_mask)[z_idx]
            center_features[idx_true[0], idx_true[1]] = self.scale

            node_mask = torch.zeros((n, 1), dtype=torch.bool, device=device)
            node_mask[-1] = True
            # append them
            prior_features["node_positions"] = torch.cat(
                [prior_features["node_positions"], center_pos], dim=0
            )
            prior_features["node_features"] = torch.cat(
                [prior_features["node_features"], center_features], dim=0
            )
            prior_features[om_diff.noise_model.node_mask_key] = node_mask
            prior_features["condition"] = F.one_hot(
                torch.as_tensor(z_idx), torch.sum(om_diff.node_features_mask).long()
            ).float()  # FIXME get name from regressor

            data = AtomsData(**prior_features)
            data_list.append(data)

        batch = collate_data(data_list)
        batch = om_diff.update_connectivity(inputs=batch, fully_connected_if_None=True)

        batch.node_features_mask = node_features_mask
        batch = batch.to(num_nodes.device)

        return batch

    @torch.no_grad()
    def generate(
            self,
            om_diff: OMDiff,
            num_samples: int = 1,
            yield_init: bool = False,
            yield_every: int = 1,
            zs_idx: Optional[torch.Tensor] = None,
    ) -> Iterator[Tuple[int, Batch]]:  # return type generator
        """Generate samples from the backward diffusion process.
        Returns:
            Generator that yields (t, samples) tuples.
        """
        device = next(om_diff.parameters()).device

        if zs_idx is None:
            (zs_idx, _), (_, num_nodes) = om_diff.num_nodes_distribution.sample(
                num_samples, return_indices=True
            )
        else:
            assert torch.numel(zs_idx) == 1 or (
                    num_samples == len(zs_idx)
            ), "If 'zs_idx' contains more than one element, its should match 'num_samples'."
            if torch.numel(zs_idx) == 1:
                zs_idx = torch.full(
                    (num_samples,),
                    fill_value=zs_idx.item(),
                    device=device,
                )

            _, num_nodes = om_diff.num_nodes_distribution.sample_conditioned(
                zs_idx, return_indices=False
            )
        yield_every = yield_every if yield_every > 0 else om_diff.timesteps
        # Sample t=T from prior
        samples = self.sample_prior(om_diff=om_diff, zs_idx=zs_idx, num_nodes=num_nodes)
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


class RegressorGuidedMaskedSampler(MaskedSampler):
    def __init__(
            self,
            guidance_strength: float,
            target_function: Callable[[Batch, torch.Tensor], torch.Tensor],
            grad_keys: Optional[list[str]] = None,
            clamp_value: float = 1.0,
            feedback_from_step: int = 1000,
            scale: float = 1.0,
    ):
        super(RegressorGuidedMaskedSampler, self).__init__(scale=scale)
        self.guidance_strength = guidance_strength
        self.target_function = target_function
        if grad_keys is None:
            grad_keys = ["node_features", "node_positions"]
        self.grad_keys = grad_keys
        self.clamp_value = clamp_value
        self.feedback_from_step = feedback_from_step

    @torch.no_grad()
    def sample_intermediate_step(
            self, om_diff: OMDiff, inputs: Batch, t: torch.LongTensor
    ) -> Batch:
        device = inputs.node_features.device
        inputs = super().sample_intermediate_step(om_diff, inputs, t)

        if t.item() <= self.feedback_from_step:
            denoised_inputs = copy.deepcopy(inputs)
            _, _, std = om_diff.noise_model.backward_coefficients(t)
            with torch.enable_grad():
                for key in self.grad_keys:
                    zs = getattr(inputs, key)
                    setattr(inputs, key, zs.requires_grad_())

                ts = torch.full(
                    (inputs.num_data, 1),
                    fill_value=t.item() - 1,
                    device=device,
                )

                energy = self.guidance_strength * self.target_function(inputs, ts)
                grad_inputs = [getattr(inputs, key) for key in self.grad_keys]
                grads = torch.autograd.grad(energy, grad_inputs, allow_unused=False)
            # handle masking, mask nodes and features
            node_mask = getattr(inputs, om_diff.noise_model.node_mask_key)
            for i, key in enumerate(self.grad_keys):
                features_mask: Union[torch.Tensor, None] = getattr(inputs, f"{key}_mask", None)
                if features_mask is None:
                    mask = node_mask
                else:
                    mask = torch.bitwise_or(features_mask, node_mask)

                total_norm = torch.linalg.vector_norm(grads[i], dim=-1, keepdim=True)

                # from PyTorch
                clip_coef = self.clamp_value / (total_norm + 1e-6)
                # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
                # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
                # when the gradients do not reside in CPU memory.
                clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

                grad = clip_coef_clamped * grads[i]
                zs = getattr(denoised_inputs, key)
                zs = torch.where(mask, zs, zs + std * grad)
                setattr(denoised_inputs, key, zs.detach())
        else:
            denoised_inputs = inputs
        denoised_inputs = om_diff.update_connectivity(denoised_inputs, fully_connected_if_None=False)

        return denoised_inputs
