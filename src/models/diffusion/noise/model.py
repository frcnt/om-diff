from abc import ABC
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.ops as ops
from src.data.components import Batch
from src.models.diffusion.noise.schedule import NoiseSchedule


class NoiseModelError(Exception):
    pass


class NoiseModel(nn.Module):
    allowed_parametrization: set

    def __init__(
            self,
            shape_mapping: dict[str, tuple[int]],
            noise_schedule: NoiseSchedule,
            parametrization: str,
    ):
        super(NoiseModel, self).__init__()
        self.shape_mapping = shape_mapping
        self.noise_schedule = noise_schedule
        if parametrization not in self.allowed_parametrization:
            raise NoiseModelError(f"Parametrization '{parametrization}' is not supported.")
        self.parametrization = parametrization

    def forward_diffusion(
            self, inputs: Batch, ts: torch.LongTensor
    ) -> Tuple[Batch, dict[str, torch.Tensor], torch.Tensor]:
        """
        Adds noise to 'inputs[self.keys]'.
        """
        raise NotImplementedError

    def backward_diffusion(
            self, inputs: Batch, predictions: dict[str, torch.Tensor], t: torch.Tensor
    ) -> Batch:
        """
        Updates 'inputs[self.keys]' based on denoising network's predictions.
        """
        raise NotImplementedError


class ContinuousNoiseModel(NoiseModel, ABC):
    def sample_noise(self, inputs: Batch) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class NormalNoiseModel(ContinuousNoiseModel):
    allowed_parametrization = ("eps",)

    def __init__(
            self,
            shape_mapping: dict[str, tuple[int]],
            noise_schedule: NoiseSchedule,
            parametrization: str = "eps",
            center_keys: Optional[list[str]] = None,
            prior_scale: Optional[dict[str, float]] = None,
    ):
        super(NormalNoiseModel, self).__init__(shape_mapping, noise_schedule, parametrization)
        if not all(k in shape_mapping for k in center_keys):
            raise NoiseModelError("'center_keys' should match keys in 'shape_mapping'.")
        self.center_keys = set() if center_keys is None else list(set(center_keys))

        self.prior_scale = nn.ParameterDict()
        for key in shape_mapping:
            if prior_scale and key in prior_scale:
                scale = torch.as_tensor(prior_scale[key])
            else:
                scale = torch.as_tensor(1.0)
            self.prior_scale[key] = nn.Parameter(scale, requires_grad=False)

    def forward_diffusion(
            self, inputs: Batch, ts: torch.LongTensor
    ) -> Tuple[Batch, dict[str, torch.Tensor], torch.Tensor]:
        # Sample noise
        noise = self.sample_noise(inputs)
        # Noise schedule
        gamma_t, alpha_t, sigma_t = self.noise_schedule.forward(ts)
        alpha_t = torch.repeat_interleave(alpha_t, inputs.num_nodes, dim=0)
        sigma_t = torch.repeat_interleave(sigma_t, inputs.num_nodes, dim=0)
        # Add noise
        for key in self.shape_mapping:
            x = getattr(inputs, key)
            x = alpha_t * x + sigma_t * noise[key]
            setattr(inputs, key, x)

        return inputs, noise, gamma_t

    def sample_prior_one(self, n: int, device: torch.device) -> dict[str, torch.Tensor]:
        sample = {}
        for key, shape in self.shape_mapping.items():
            sample_key = torch.randn((n, *shape), device=device) * self.prior_scale[key].to(device)
            if key in self.center_keys:
                sample_key -= sample_key.mean(dim=0)
            sample[key] = sample_key
        return sample

    def sample_final_step(self, inputs: Batch, predictions: dict[str, torch.Tensor]):
        device = inputs.node_features.device
        _, alpha_0, sigma_0 = self.noise_schedule(
            torch.tensor([0], dtype=torch.long, device=device)
        )
        noise = self.sample_noise(inputs)
        for key in self.shape_mapping:
            phi_zt = predictions[key]
            zt = getattr(inputs, key)
            eps = noise[key]

            mean_zs = (zt - sigma_0 * phi_zt) / alpha_0
            std = sigma_0 / alpha_0
            zs = mean_zs + std * eps
            if key in self.center_keys:
                zs = ops.center_splits(zs, inputs.num_nodes)

            setattr(inputs, key, zs)

        return inputs

    def backward_coefficients(self, t: torch.LongTensor):
        gamma_t, _, sigma_t = self.noise_schedule.forward(t)
        gamma_s, _, sigma_s = self.noise_schedule.forward(t - 1)

        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        prefactor = sigma2_t_given_s / (alpha_t_given_s * sigma_t)
        std = sigma_t_given_s * sigma_s / sigma_t

        return alpha_t_given_s, prefactor, std

    def backward_diffusion(
            self, inputs: Batch, predictions: dict[str, torch.Tensor], t: torch.LongTensor
    ) -> Batch:
        alpha_t_given_s, prefactor, std = self.backward_coefficients(t)
        noise = self.sample_noise(inputs)
        for key in self.shape_mapping:
            phi_zt = predictions[key]
            zt = getattr(inputs, key)
            eps = noise[key]

            mean_zs = zt / alpha_t_given_s - prefactor * phi_zt
            zs = mean_zs + std * eps
            if key in self.center_keys:
                zs = ops.center_splits(zs, inputs.num_nodes)

            setattr(inputs, key, zs)
        return inputs

    def sample_noise(self, inputs: Batch) -> dict[str, torch.Tensor]:
        noise = {}
        for key in self.shape_mapping:
            target = getattr(inputs, key)
            noise_key = torch.randn_like(target) * self.prior_scale[key]

            if key in self.center_keys:
                noise_key = ops.center_splits(noise_key, inputs.num_nodes)

            noise[key] = noise_key
        return noise


class MaskedNormalNoiseModel(NormalNoiseModel):
    def __init__(
            self,
            shape_mapping: dict[str, tuple[int]],
            noise_schedule: NoiseSchedule,
            node_mask_key: str = "node_mask",
            parametrization: str = "eps",
            center_keys: Optional[list[str]] = None,
            prior_scale: Optional[dict[str, float]] = None,
    ):
        super(MaskedNormalNoiseModel, self).__init__(
            shape_mapping=shape_mapping,
            noise_schedule=noise_schedule,
            parametrization=parametrization,
            center_keys=center_keys,
            prior_scale=prior_scale,
        )
        self.node_mask_key = node_mask_key

    def sample_prior_one(
            self, n: int, device: torch.device, masks: Optional[dict[str, torch.Tensor]] = None
    ) -> dict[str, torch.Tensor]:
        masks = {} if masks is None else masks
        sample = {}
        for key, shape in self.shape_mapping.items():
            sample_key = self.sample_masked_noise(
                key, shape=(n, *shape), device=device, mask=masks.get(key, None)
            )  # NOTE: masking should be handled outside as 'mask' argument is not used.
            if key in self.center_keys:
                sample_key -= sample_key.mean(dim=0)
            sample[key] = sample_key
        return sample

    def sample_noise(self, inputs: Batch) -> dict[str, torch.Tensor]:
        noise = {}
        node_mask = getattr(inputs, self.node_mask_key)
        for key in self.shape_mapping:
            target: torch.Tensor = getattr(inputs, key)
            features_mask: Union[torch.Tensor, None] = getattr(inputs, f"{key}_mask", None)
            if features_mask is None:
                mask = node_mask
            else:
                mask = torch.bitwise_or(features_mask, node_mask)
            noise_key = self.sample_masked_noise(
                key=key, mask=mask, shape=target.shape, device=target.device
            )
            if key in self.center_keys:
                noise_key = ops.center_splits_with_mask(noise_key, inputs.num_nodes, mask)

            noise[key] = noise_key
        return noise

    def sample_masked_noise(
            self,
            key: str,
            shape: tuple[int],
            device: torch.device,
            mask: Optional[torch.Tensor] = None,
    ):
        noise = torch.randn(shape, device=device) * self.prior_scale[key].to(device)
        if mask is not None:
            noise = torch.where(mask, torch.zeros(shape, device=device), noise)

        return noise

    def forward_diffusion(
            self, inputs: Batch, ts: torch.LongTensor
    ) -> Tuple[Batch, dict[str, torch.Tensor], torch.Tensor]:
        # Sample noise
        noise = self.sample_noise(inputs)
        # Noise schedule
        gamma_t, alpha_t, sigma_t = self.noise_schedule.forward(ts)
        alpha_t = torch.repeat_interleave(alpha_t, inputs.num_nodes, dim=0)
        sigma_t = torch.repeat_interleave(sigma_t, inputs.num_nodes, dim=0)
        # Add masked noise
        node_mask = getattr(inputs, self.node_mask_key)
        for key in self.shape_mapping:
            x = getattr(inputs, key)
            x = torch.where(node_mask, x, alpha_t * x + sigma_t * noise[key])
            setattr(inputs, key, x)

        return inputs, noise, gamma_t

    def backward_diffusion(
            self, inputs: Batch, predictions: dict[str, torch.Tensor], t: torch.LongTensor
    ) -> Batch:
        alpha_t_given_s, prefactor, std = self.backward_coefficients(t)

        noise = self.sample_noise(inputs)

        node_mask = getattr(inputs, self.node_mask_key)
        for key in self.shape_mapping:
            phi_zt = predictions[key]
            zt = getattr(inputs, key)
            eps = noise[key]

            mean_zs = zt / alpha_t_given_s - prefactor * phi_zt
            zs = torch.where(node_mask, zt, mean_zs + std * eps)

            if key in self.center_keys:
                zs = ops.center_splits_with_mask(zs, inputs.num_nodes, mask=inputs.node_mask)

            setattr(inputs, key, zs)
        return inputs

    def sample_final_step(self, inputs: Batch, predictions: dict[str, torch.Tensor]):
        device = inputs.node_features.device
        _, alpha_0, sigma_0 = self.noise_schedule(
            torch.tensor([0], dtype=torch.long, device=device)
        )
        noise = self.sample_noise(inputs)
        node_mask = getattr(inputs, self.node_mask_key)
        for key in self.shape_mapping:
            phi_zt = predictions[key]
            zt = getattr(inputs, key)
            eps = noise[key]

            mean_zs = (zt - sigma_0 * phi_zt) / alpha_0
            std = sigma_0 / alpha_0
            zs = torch.where(node_mask, zt, mean_zs + std * eps)
            if key in self.center_keys:
                zs = ops.center_splits_with_mask(zs, inputs.num_nodes, mask=inputs.node_mask)

            setattr(inputs, key, zs)

        return inputs
