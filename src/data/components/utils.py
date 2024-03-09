from typing import Optional
from typing import Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from src.data.components.datasets.base import BaseDataset
from src.data.utils import read_json


def split_train_val_test(
    dataset: BaseDataset,
    train_val_split: Tuple[float, float] = (0.8, 0.1),
    bootstrap: bool = False,
    shuffle: bool = True,
    split_seed: int = 42,
    only_idx: bool = False,
) -> Union[dict[str, np.ndarray], dict[str, BaseDataset]]:
    rnd = np.random.RandomState(seed=split_seed)

    n_samples = len(dataset)
    n_train, n_val = int(n_samples * train_val_split[0]), int(n_samples * train_val_split[1])
    assert n_train + n_val <= n_samples

    idx = np.arange(n_samples)

    if shuffle:
        idx = rnd.permutation(idx)

    train_idx = rnd.choice(idx[:n_train], n_train) if bootstrap else idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    if only_idx:
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    return {
        "train": dataset.create_subset(train_idx),
        "val": dataset.create_subset(val_idx),
        "test": dataset.create_subset(test_idx),
    }


def split_from_file(
    dataset: BaseDataset,
    split_file: str,
    split_seed: int = 42,
    train_prop: float = 0.95,
    bootstrap: bool = False,
    shuffle: bool = True,
) -> dict[str, BaseDataset]:
    split_dict = read_json(split_file)
    assert all(k in split_dict for k in ["train", "test"])
    if "val" not in split_dict:
        rnd = np.random.RandomState(seed=split_seed)

        train_val_idx = np.array(split_dict["train"])
        n_train_val = len(train_val_idx)
        idx = np.arange(n_train_val)

        n_train = int(n_train_val * train_prop)
        if shuffle:
            idx = rnd.permutation(idx)
        split_dict["train"] = (
            rnd.choice(train_val_idx[idx[:n_train]], n_train)
            if bootstrap
            else train_val_idx[idx[:n_train]]
        )
        split_dict["val"] = train_val_idx[idx[n_train:]]
    return {fold: dataset.create_subset(split_dict[fold]) for fold in split_dict}


def adjacency_matrix_from_edge_index(
    edge_index: torch.Tensor, num_nodes: Optional[int] = None
) -> torch.Tensor:
    """Create adjacency matrix from edge index.

    Args:
        edge_index (tensor): Tensor with shape (num_edges, 2).
        num_nodes (int): Number of nodes.
    Returns:
        Adjacency matrix tensor with shape (num_nodes, num_nodes).
    """
    assert edge_index.shape[1] == 2
    assert num_nodes is None or num_nodes > edge_index.max()
    num_nodes = num_nodes if num_nodes is not None else edge_index.max() + 1
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    A[edge_index[:, 0], edge_index[:, 1]] = 1
    return A


def edge_index_from_adjacency_matrix(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """Create edge index from adjacency matrix.

    Args:
        adjacency_matrix (tensor): Tensor with shape (num_nodes, num_nodes).
    Returns:
        Edge index tensor with shape (num_edges, 2).
    """
    edge_index = torch.argwhere(adjacency_matrix).contiguous()
    return edge_index


def progress_hook(t: tqdm):
    """
    Callback displaying progress during download.
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to
