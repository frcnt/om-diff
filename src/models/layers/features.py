import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.data.components import Batch


class OneHotEmbedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            input_key: str = "node_features",
            output_key: str = None,
    ):
        super().__init__()
        self.embedding = nn.Linear(num_embeddings, embedding_dim, bias=False)
        self.input_key = input_key
        self.output_key = input_key if output_key is None else output_key

    def forward(self, inputs: Batch) -> Batch:
        x = getattr(inputs, self.input_key)
        x = self.embedding(x)

        setattr(inputs, self.output_key, x)

        return inputs


class Combine(nn.Module):
    def __init__(
            self,
            input_keys: list[str],
            output_key: str,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_keys = input_keys
        self.output_key = output_key

    def forward(self, inputs: Batch) -> Batch:
        xs = [getattr(inputs, key) for key in self.input_keys]
        x = torch.cat(xs, dim=-1)
        setattr(inputs, self.output_key, x)
        return inputs


class ScaleShift(nn.Module):
    def __init__(
            self,
            input_scales: dict[str, Union[float, list[float]]] = None,
            input_shifts: dict[str, Union[float, list[float]]] = None,
            output_keys: Optional[list[str]] = None,
            trainable: bool = False,
            **kwargs,
    ):
        """

        Args:
            input_scales (dict[str, float]): mapping input_key to scale
            input_scales (dict[str, float]): mapping input_key to shift
            output_keys (list[str], optional):
            **kwargs:
        """
        super().__init__(**kwargs)
        assert set(input_scales.keys()) == set(input_shifts.keys())

        self.input_scales = nn.ParameterDict(
            {
                key: nn.Parameter(torch.as_tensor(input_scales[key]), requires_grad=not trainable)
                for key in sorted(input_scales)
            },
        )
        self.input_shifts = nn.ParameterDict(
            {
                key: nn.Parameter(torch.as_tensor(input_shifts[key]), requires_grad=not trainable)
                for key in sorted(input_shifts)
            },
        )
        if output_keys is None:
            self.output_keys = list(input_scales.keys())
        else:
            assert len(output_keys) == len(input_scales)
            self.output_keys = output_keys

    def forward(self, inputs: Batch) -> Batch:
        for input_key, output_key in zip(self.input_scales, self.output_keys):
            x = (
                    getattr(inputs, input_key) * self.input_scales[input_key]
                    + self.input_shifts[input_key]
            )
            setattr(inputs, output_key, x)
        return inputs


class ConditionalScaleShift(nn.Module):
    """
    Scales and shifts specified features by specified values.
    """

    def __init__(
            self,
            condition_key: str,
            input_scales: dict[str, Union[float, list[float]]] = None,
            input_shifts: dict[str, Union[float, list[float]]] = None,
            output_keys: Optional[list[str]] = None,
            trainable: bool = False,
            **kwargs,
    ):
        """

        Args:
            input_scales (dict[str, float]): mapping input_key to scale
            input_scales (dict[str, float]): mapping input_key to shift
            output_keys (list[str], optional):
            **kwargs:
        """
        super().__init__(**kwargs)
        assert set(input_scales.keys()) == set(input_shifts.keys())

        self.condition_key = condition_key

        self.input_scales = nn.ParameterDict(
            {
                key: nn.Parameter(torch.as_tensor(input_scales[key]), requires_grad=not trainable)
                for key in sorted(input_scales)
            },
        )
        self.input_shifts = nn.ParameterDict(
            {
                key: nn.Parameter(torch.as_tensor(input_shifts[key]), requires_grad=not trainable)
                for key in sorted(input_shifts)
            },
        )
        if output_keys is None:
            self.output_keys = list(input_scales.keys())
        else:
            assert len(output_keys) == len(input_scales)
            self.output_keys = output_keys

    def forward(self, inputs: Batch) -> Batch:
        condition = getattr(inputs, self.condition_key)

        for input_key, output_key in zip(self.input_scales, self.output_keys):
            scale = torch.sum(self.input_scales[input_key] * condition, dim=-1, keepdim=True)
            shift = torch.sum(self.input_shifts[input_key] * condition, dim=-1, keepdim=True)

            x = getattr(inputs, input_key) * scale + shift
            setattr(inputs, output_key, x)
        return inputs


class FourierFeatures(nn.Module):
    """
    Random Fourier features (sine and cosine expansion).
    OBS: The dimensionality of the output is 2 * output_features.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            input_key: str,
            output_key: Optional[str] = None,
            std: float = 1.0,
            trainable: bool = False,
    ):
        super(FourierFeatures, self).__init__()
        weight = torch.normal(mean=torch.zeros(out_features, in_features), std=std)

        self.trainable = trainable
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

        self.input_key = input_key
        self.output_key = input_key if output_key is None else output_key

    def forward(self, inputs: Batch):
        # Modulate the input by the different frequencies
        x = getattr(inputs, self.input_key)
        x = F.linear(x, self.weight)
        # Calculate cosine and sine
        cos_features = torch.cos(2 * math.pi * x)
        sin_features = torch.sin(2 * math.pi * x)
        # Concatenate sine and cosine features
        x = torch.cat((cos_features, sin_features), dim=1)
        setattr(inputs, self.output_key, x)

        return inputs


class PairwiseLayer(nn.Module):
    """
    Dummy layer computing pairwise distances and vectors.
    """

    def __init__(self, norm: bool = True):
        """
        :param norm: bool (default: True), whether the pairwise vectors should be normed
        """
        super().__init__()
        self.norm = norm

    def forward(self, positions: torch.Tensor, bonds: torch.Tensor, norm: bool = None):
        norm = self.norm if norm is None else norm

        vectors = (
                positions[bonds[:, 1], :] - positions[bonds[:, 0], :]
        )  # (n_bonds, 3) vector (i - > j)
        distances = torch.sqrt(
            torch.sum(vectors ** 2, dim=-1, keepdim=True) + 1e-6
        )  # (n_bonds, 1)

        if norm:
            vectors = vectors / distances

        return distances, vectors  # (n_bonds, 1), (n_bonds, 3)
