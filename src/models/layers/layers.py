import math
from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_min, scatter_max
from torch_scatter.composite import scatter_softmax, scatter_std

import src.models.ops as ops
from src.data.components import Batch


class IndexEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        input_key: str = "node_features",
        output_key: str = None,
        **kwargs,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.input_key = input_key
        self.output_key = input_key if output_key is None else output_key

    def forward(self, inputs: Batch) -> Batch:
        x = getattr(inputs, self.input_key)
        if len(x.shape) > 1:
            assert len(x.shape) == 2
            assert x.shape[1] == self.num_embeddings
            x = torch.argmax(x, dim=-1)
        x = super(IndexEmbedding, self).forward(x)
        setattr(inputs, self.output_key, x)

        return inputs


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
    """
    Simple module that combines features to a new, user-specified, feature.
    """

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
        # TODO: handles shapes properly, e.g. check all but the last are identical
        xs = [getattr(inputs, key) for key in self.input_keys]
        x = torch.cat(xs, dim=-1)
        setattr(inputs, self.output_key, x)
        return inputs


class ScaleShift(nn.Module):
    """
    Scales and shifts specified features by specified values.
    """

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


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_hidden_layers: int = 1,
        activation_cls=nn.SiLU,
    ):
        super().__init__()
        assert num_hidden_layers >= 0
        self.input_dim = input_dim
        self.output_dim = output_dim

        if num_hidden_layers == 0:
            self.nn = nn.Linear(input_dim, output_dim)
        else:
            assert hidden_dim is not None
            self.nn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_cls(),
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation_cls())
                    for _ in range(num_hidden_layers - 1)
                ],
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x):
        return self.nn(x)


class Readout(nn.Module):
    """
    Generic Readout Layer.
    """

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

        # c = ops.sum_splits(torch.amax(s.detach().clone(), dim=-1, keepdim=True), inputs.num_nodes)
        # c = torch.repeat_interleave(c, inputs.num_nodes, dim=0)

        # s = torch.exp(s - c)
        # s = torch.exp(s)
        # s_norm =
        # unweighted_x = ops.sum_splits(s * x, splits)

        # x = unweighted_x / (s_norm + 1e-6)

        sx = ops.sum_splits(s * x, splits)
        out = self.readout_net(sx)

        setattr(inputs, self.output_key, out)
        return inputs


class PNAReadout(nn.Module):
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

        std_x = scatter_std(x, index=inputs.index[:, None], dim=0)
        min_x, _ = scatter_min(x, index=inputs.index[:, None], dim=0)
        max_x, _ = scatter_max(x, index=inputs.index[:, None], dim=0)

        y = torch.cat([sx, std_x, min_x, max_x], dim=1)

        out = self.readout_net(y)

        setattr(inputs, self.output_key, out)
        return inputs


class NegativeGradient(nn.Module):
    def __init__(
        self,
        energy_key: str,
        wrt_key: str,
        output_key: str,
    ):
        super(NegativeGradient, self).__init__()
        self.energy_key = energy_key
        self.wrt_key = wrt_key
        self.output_key = output_key

        self.model_outputs = [self.output_key]
        self.required_derivatives = [self.wrt_key]

    def forward(self, inputs: Batch) -> Batch:
        energy_pred = getattr(inputs, self.energy_key)

        go: List[Optional[torch.Tensor]] = [torch.ones_like(energy_pred)]
        grads = torch.autograd.grad(
            [energy_pred],
            [getattr(inputs, prop) for prop in self.required_derivatives],
            grad_outputs=go,
            create_graph=self.training,
        )
        grad = -grads[0]

        setattr(inputs, self.output_key, grad)
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


class PositiveLinear(nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, weight_init_offset: int = -2
    ):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


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
            torch.sum(vectors**2, dim=-1, keepdim=True) + 1e-6
        )  # (n_bonds, 1)

        if norm:
            vectors = vectors / distances

        return distances, vectors  # (n_bonds, 1), (n_bonds, 3)


class EquivariantLayerNorm(nn.Module):
    r"""Rotationally-equivariant Vector Layer Normalization
    Expects inputs with shape (N, n, d), where N is batch size, n is vector dimension, d is width/number of vectors.
    """
    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EquivariantLayerNorm, self).__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)  # Without bias term to preserve equivariance!

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input):
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input):
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix):
        """Compute the inverse square root of a positive definite matrix.
        Based on https://github.com/pytorch/pytorch/issues/25481
        """
        _, s, v = matrix.svd()
        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(-2, -1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.float64)  # Need double precision for accurate inversion.
        input = self.mean_center(input)
        # We use different diagonal elements in case input matrix is approximately zero,
        # in which case all singular values are equal which is problematic for backprop.
        # See e.g. https://pytorch.org/docs/stable/generated/torch.svd.html
        reg_matrix = (
            torch.diag(torch.tensor([1.0, 2.0, 3.0]))
            .unsqueeze(0)
            .to(input.device)
            .type(input.dtype)
        )
        covar = self.covariance(input) + self.eps * reg_matrix
        covar_sqrtinv = self.symsqrtinv(covar)
        return (covar_sqrtinv @ input).to(self.weight.dtype) * self.weight.reshape(
            1, 1, self.normalized_shape[0]
        )

    def extra_repr(self) -> str:
        return "{normalized_shape}, " "elementwise_linear={elementwise_linear}".format(
            **self.__dict__
        )


class GraphLayerNorm(torch.nn.Module):
    """
    Layer normalisation on a per graph basis.
    """

    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(in_channels))
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
        x = ops.center_splits(x - x.mean(-1, keepdim=True), splits=splits)
        var_x = torch.mean(ops.mean_splits(x * x, splits=splits), dim=-1, keepdim=True)
        var_x = torch.repeat_interleave(var_x, splits, dim=0)

        out = x / torch.sqrt(var_x + self.eps)

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out


class GraphLayerNormImproved(nn.Module):
    """
    Layer normalisation on a per graph basis.
    """

    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(in_channels))
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(
        self, s: torch.Tensor, v: torch.Tensor, splits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # scalar
        centered_s = ops.center_splits(s - s.mean(-1, keepdim=True), splits=splits)
        var_s = torch.mean(
            ops.mean_splits(centered_s * centered_s, splits=splits), dim=-1, keepdim=True
        )
        var_s = torch.sqrt(torch.repeat_interleave(var_s, splits, dim=0) + self.eps)

        sout = centered_s / var_s
        if self.weight is not None and self.bias is not None:
            sout = sout * self.weight + self.bias

        # vector
        vnorm = torch.mean(
            torch.sqrt(torch.sum(v**2, dim=1) + self.eps),
            dim=-1,
            keepdim=True,
        )
        vnorm = ops.mean_splits(vnorm, splits)
        vnorm = torch.repeat_interleave(vnorm, splits, dim=0)
        vout = v / vnorm[..., None]

        return sout, vout
