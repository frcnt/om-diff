import math

import torch
import torch.nn as nn


def envelop_fn(
    x: torch.Tensor,
    xc: float,
    p: int,
):
    x_norm = x / xc
    x_p = x_norm**p
    x_p1 = x_p * x_norm

    x_p2 = x_p1 * x_norm
    return torch.where(
        x < xc,
        1.0 - x_p * (p + 1) * (p + 2) / 2.0 + p * (p + 2) * x_p1 - p * (p + 1) * x_p2 / 2.0,
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
    )


class EnvelopLayer(nn.Module):
    def __init__(self, p: int = 6, xc: float = 5.0):
        super().__init__()
        self.register_buffer("p", torch.LongTensor([p]))
        self.register_buffer("xc", torch.FloatTensor([xc]))

    def forward(self, distances: torch.Tensor):
        return envelop_fn(distances, xc=self.xc, p=self.p)


class RBFLayer(nn.Module):
    n_features: int

    def __init__(self, n_features: int):
        super(RBFLayer, self).__init__()
        self.n_features = n_features


class BesselRBFLayer(RBFLayer):
    def __init__(self, n_features: int = 20, max_distance: float = 5.0, trainable: bool = True):
        super(BesselRBFLayer, self).__init__(n_features)

        self.trainable = trainable

        self.r_max = float(max_distance)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = torch.linspace(start=1.0, end=n_features, steps=n_features) * math.pi
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights[None, :] * distances / self.r_max)

        return self.prefactor * (numerator / distances)


class GaussianLinearRBFLayer(RBFLayer):
    def __init__(
        self,
        n_features: int = 150,
        max_distance: float = 5.0,
        min_distance: float = 0.0,
    ):
        super().__init__(n_features)
        self.register_buffer("delta", torch.tensor((max_distance - min_distance) / n_features))
        self.register_buffer(
            "offsets", torch.linspace(start=min_distance, end=max_distance, steps=n_features)
        )

    def forward(self, distances: torch.Tensor):
        return torch.exp((-((distances - self.offsets[None, :]) ** 2)) / self.delta)
