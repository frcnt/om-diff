from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def clip_noise_schedule(alphas2: np.ndarray, clip_value: float = 0.001):
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


class NoiseSchedule(nn.Module):
    timesteps: int

    def forward_gamma(self, t: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
            self,
            t: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert t.min() >= 0 and t.max() <= self.timesteps

        gamma_t = self.forward_gamma(t)
        alpha_t, sigma_t = self.get_alpha_sigma_from_gamma(gamma_t)

        return gamma_t, alpha_t, sigma_t

    @staticmethod
    def get_alpha_sigma_from_gamma(gamma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha2 = torch.sigmoid(-gamma)
        sigma2 = 1 - alpha2  # sigmoid identity S(x) = 1 - S(-x)
        return torch.sqrt(alpha2), torch.sqrt(sigma2)


class FixedNoiseSchedule(NoiseSchedule):
    def __init__(self, timesteps: int, precision: float = 1e-4):
        super().__init__()
        self.timesteps = timesteps
        # Compute gamma(t) from alpha_t^2 and sigma_t^2
        # alpha and sigma can later be computed from gamma.
        ts = torch.arange(0, timesteps + 1, dtype=torch.float)
        alpha2 = torch.square((1 - 2 * precision) * (1 - torch.square(ts / timesteps)) + precision)
        sigma2 = 1 - alpha2
        gamma = -(torch.log(alpha2) - torch.log(sigma2))

        self.register_buffer("gamma", gamma)

    def forward_gamma(self, t: torch.LongTensor):
        return self.gamma[t]


class PolynomialNoiseSchedule(FixedNoiseSchedule):
    def __init__(self, timesteps: int, precision: float = 1e-4, power: float = 2):
        super(PolynomialNoiseSchedule, self).__init__(timesteps)
        self.timesteps = timesteps

        alphas2 = self.schedule(timesteps, precision=precision, power=power)
        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.register_buffer("gamma", torch.from_numpy(-log_alphas2_to_sigmas2).float())

    @staticmethod
    def schedule(timesteps: int, precision: float = 1e-4, power: float = 3.0):
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas2 = (1 - np.power(x / steps, power)) ** 2

        alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

        precision = 1 - 2 * precision

        alphas2 = precision * alphas2 + precision

        return alphas2
