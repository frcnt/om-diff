import warnings
from typing import Optional, Union

import torch


def sum_splits(
    values: torch.Tensor,
    splits: Union[torch.Tensor, torch.LongTensor],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum values in splits.

    Args:
        values (tensor): The input values to be reduced with shape (num_values, ...).
        splits (tensor(int)): The split sizes with shape (num_splits,) and sum(splits) = num_values.
        out (tensor): The tensor used to store the output with shape (num_splits, ...).
    Returns:
        A tensor with size (num_splits, ...).
    """
    index = torch.repeat_interleave(
        torch.arange(splits.shape[0], device=values.device, dtype=torch.long), splits
    )
    return sum_index(values, index, out)


def mean_splits(
    values: torch.Tensor,
    splits: Union[torch.Tensor, torch.LongTensor],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean values in splits.

     Args:
        values (tensor): The input values to be reduced with shape (num_values, ...).
        splits (tensor(int)): The split sizes with shape (num_splits,) and sum(splits) = num_values.
        out (tensor): The tensor used to store the output with shape (num_splits, ...).
    Returns:
        A tensor with size (num_splits, ...).
    """
    assert (
        values.ndim >= 2
    )  # values must be at least 2D because of division by splits.unsqueeze(1)
    return sum_splits(values, splits, out=out) / splits.unsqueeze(1)


def reduce_splits(
    values: torch.Tensor,
    splits: Union[torch.Tensor, torch.LongTensor],
    out: Optional[torch.Tensor] = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Reduce values in splits.

    Can be used to reduce from node-level values to graph-level values in a batch
    when values are node_states and splits is a tensor of num_nodes.

     Args:
        values (tensor): The input values to be reduced with shape (num_values, ...).
        splits (tensor(int)): The split sizes with shape (num_splits,) and sum(splits) = num_values.
        out (tensor): The tensor used to store the output with shape (num_splits, ...).
        reduction (string): The reduction method.
    Returns:
        A tensor with size (num_splits, ...).
    """
    if reduction == "sum":
        return sum_splits(values, splits, out=out)
    elif reduction == "mean":
        return mean_splits(values, splits, out=out)
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")


def center_splits(
    values: torch.Tensor,
    splits: Union[torch.Tensor, torch.LongTensor],
):
    """Center values in splits to zero mean.

    Args:
        values (tensor): The input values to be centered (num_values, ...).
        splits (tensor(int)): The split sizes with shape (num_splits,) and sum(splits) = num_values.
    """
    means = mean_splits(values, splits)
    means = torch.repeat_interleave(means, splits, dim=0)
    centered_values = values - means
    if centered_values.mean() > 5e-2:
        warnings.warn(f"After centering, the mean value is still larger than '5e-2'.")
    return centered_values


def center_splits_with_mask(
    values: torch.Tensor,
    splits: Union[torch.Tensor, torch.LongTensor],
    mask: torch.Tensor,
):
    """Center values in splits to zero mean.

    Args:
        values (tensor): The input values to be centered (num_values, ...).
        splits (tensor(int)): The split sizes with shape (num_splits,) and sum(splits) = num_values.
    """
    non_mask = (~mask).float()
    corrected_splits = splits - sum_splits(mask.long(), splits).view_as(splits)
    means = sum_splits(values * non_mask, splits) / corrected_splits.unsqueeze(1)
    means = torch.where(
        mask, torch.zeros_like(values), torch.repeat_interleave(means, splits, dim=0)
    )
    centered_values = values - means
    if centered_values.mean() > 5e-2:
        warnings.warn(f"After centering, the mean value is still larger than '5e-2'.")
    return centered_values


def sum_index(
    values: torch.Tensor,
    index: Union[torch.LongTensor, torch.Tensor],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sum values by index.

    OBS: If 'out' is not specified, its shape is inferred.
    This can lead to errors when a node is isolated in the graph, i.e. has no neighbour.
    Args:
        values (tensor): The input values to be reduced with shape (num_values, ...).
        index (tensor(int)): The indices in the output each value is reduced to (num_values,).
        out (tensor): The tensor used to accumulate the summed values, with shape (max(index), ...).
    Returns:
        A tensor with size (max(index), ...).
    """
    if out is None:
        out_shape = torch.Size([index.max() + 1]) + values.shape[1:]
        out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    out.index_put_((index,), values, accumulate=True)
    return out


def mean_index(
    values: torch.Tensor,
    index: Union[torch.LongTensor, torch.Tensor],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Average values by index.

    OBS: If 'out' is not specified, its shape is inferred.
    This can lead to errors when a node is isolated in the graph, i.e. has no neighbour.


    Args:
        values (tensor): The input values to be reduced with shape (num_values, ...).
        index (tensor(int)): The indices in the output each value is reduced to (num_values,).
        out (tensor): The tensor used to store the output with shape (max(index), ...).
    Returns:
        A tensor with size (max(index), ...).
    """
    sum_out = sum_index(values, index, out)

    in_shape = (values.shape[0],) + (1,) * (values.ndim - 1)
    out_shape = (sum_out.shape[0],) + (1,) * (sum_out.ndim - 1)

    counts = sum_index(
        torch.ones(in_shape, device=values.device),
        index,
        out=torch.zeros(out_shape, device=values.device),
    )
    counts = torch.where(counts == 0.0, torch.tensor(1.0, device=counts.device), counts)
    return sum_out / counts


def reduce_index(
    values: torch.Tensor,
    index: Union[torch.LongTensor, torch.Tensor],
    out: Optional[torch.Tensor] = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Reduce values by index.

    OBS: If 'out' is not specified, its shape is inferred.
    This can lead to errors when a node is isolated in the graph, i.e. has no neighbour.


    Args:
        values (tensor): The input values to be reduced with shape (num_values, ...).
        index (tensor(int)): The indices in the output each value is reduced to (num_values,).
        out (tensor): The tensor used to store the output with shape (max(index), ...).
        reduction (string): The reduction method.
    Returns:
        A tensor with size (max(index), ...).
    """
    if reduction == "sum":
        return sum_index(values, index, out)
    elif reduction == "mean":
        return mean_index(values, index, out)
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")
