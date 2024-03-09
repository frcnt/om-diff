import copy
from abc import abstractmethod

import numpy as np
import torch.utils.data as torchdata

from src.data.components.transforms.base import Transform


class BaseDataset(torchdata.Dataset):
    idx_subset: None

    def __init__(
        self, idx_subset: np.ndarray = None, transform: Transform = lambda _x: _x, **kwargs
    ):
        super().__init__(**kwargs)
        self.idx_subset = idx_subset
        self.transform = transform

    def convert_idx(self, idx):
        idx = np.array(idx)
        return idx if self.idx_subset is None else self.idx_subset[idx]

    def __len__(self):
        if self.idx_subset is not None:
            length = len(self.idx_subset)
        else:
            length = self._len()
        return length

    def __getitem__(self, idx, **kwargs):
        # convert in case of subset
        inner_idx = self.convert_idx(idx)
        return self._get_item(inner_idx, **kwargs)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def _len(self):
        raise NotImplementedError

    @abstractmethod
    def _get_item(self, inner_idx, **kwargs):
        raise NotImplementedError

    def create_subset(self, idx_subset, **kwargs):
        assert idx_subset is not None, "'idx_subset' is required"
        self_copy = copy.copy(self)
        idx_subset = self.convert_idx(idx_subset)
        self_copy.idx_subset = idx_subset
        return self_copy

    @property
    def node_count(self) -> dict[int, int]:
        raise NotImplementedError

    @property
    def conditional_node_count(self) -> tuple[dict[int, int], dict[int, dict[int, int]]]:
        raise NotImplementedError

    @property
    def node_label_count(self) -> dict[int, int]:
        raise NotImplementedError

    @property
    def node_labels(self) -> list[int]:
        return sorted(list(self.node_label_count))

    @property
    def num_dimensions(self) -> int:
        raise NotImplementedError
