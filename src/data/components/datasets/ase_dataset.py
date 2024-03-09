from collections import Counter
from typing import Optional

import ase
import ase.io as ase_io
import numpy as np
from ase.db import connect
from ase.db.core import Database

from src.data.components.datasets.base import BaseDataset
from src.data.components.transforms.base import Transform

__all__ = ["ASEDBDataset"]


class ASEDBDataset(BaseDataset):
    """
    PyTorch Dataset to interact with ASE database.
    """

    def __init__(
            self,
            db_path,
            transform: Transform = lambda _x: _x,
            idx_subset: np.ndarray = None,
            in_memory: bool = True,
            **kwargs,
    ):
        super().__init__(idx_subset=idx_subset, transform=transform, **kwargs)
        self.db_path = db_path
        self.db_connection: Database = connect(db_path)

        self.in_memory = in_memory
        self.cached_rows = {}
        self.populated_cache = False
        if in_memory:
            self._populate_cache()

        self._node_count = None
        self._node_label_count = None
        self._conditional_node_count = (None, None)

    def _len(self):
        return (
            len(self.db_connection)
            if not self.in_memory or not self.populated_cache
            else len(self.cached_rows)
        )

    def _get_item(self, inner_idx, transform: bool = True):
        inner_idx = int(self.to_db_idx(inner_idx))
        try:
            if self.in_memory:
                row = self.cached_rows[inner_idx]
            else:
                row = self.db_connection[inner_idx]
        except KeyError:
            raise IndexError("index out of range")

        if transform and self.transform is not None:
            return self.transform(row)
        else:
            return row

    def _populate_cache(self):
        with connect(self.db_path) as db:
            for outer_idx in range(len(self)):
                inner_idx = int(self.to_db_idx(self.convert_idx(outer_idx)))
                self.cached_rows[inner_idx] = db[inner_idx]
        self.populated_cache = True

    @staticmethod
    def to_db_idx(idx):
        # Note that ASE DB is 1-indexed
        return idx + 1

    @property
    def node_count(self) -> dict[int, int]:
        if self._node_count is None:
            counter = Counter(
                self.__getitem__(i, transform=False).natoms for i in range(len(self))
            )
            counter = dict(sorted(counter.items()))
            self._node_count = counter
        return self._node_count

    @property
    def node_label_count(self) -> dict[int, int]:
        if self._node_label_count is None:
            counter = Counter(
                n for i in range(len(self)) for n in self.__getitem__(i, transform=False).numbers
            )
            counter = dict(sorted(counter.items()))
            self._node_label_count = counter
        return self._node_label_count

    @property
    def conditional_node_count(self) -> tuple[dict[int, int], dict[int, dict[int, int]]]:
        if self._conditional_node_count == (None, None):
            node_label_count, node_count = {}, {}
            for i in range(len(self)):
                item = self.__getitem__(i, transform=False)
                numbers, natoms = set(item.numbers), item.natoms
                for n in numbers:
                    if n not in node_label_count:
                        node_label_count[n] = 0
                    node_label_count[n] += 1

                    if n not in node_count:
                        node_count[n] = {}

                    if natoms not in node_count[n]:
                        node_count[n][natoms] = 0

                    node_count[n][natoms] += 1

            node_label_count = dict(sorted(node_label_count.items()))
            node_count = dict(sorted(node_count.items()))
            for k in node_count:
                node_count[k] = dict(sorted(node_count[k].items()))

            self._conditional_node_count = node_label_count, node_count

        return self._conditional_node_count

    @property
    def num_dimensions(self) -> int:
        return 3


class XYZDataset(BaseDataset):
    """
    PyTorch Dataset to interact with XYZ file.
    Always in-memory.
    """

    def __init__(
            self,
            xyz_path: Optional[str] = None,
            transform: Transform = lambda _x: _x,
            idx_subset: np.ndarray = None,
            atoms_lst: Optional[list[ase.Atoms]] = None,
            **kwargs,
    ):
        super().__init__(idx_subset=idx_subset, transform=transform, **kwargs)
        if xyz_path is None:
            assert atoms_lst is not None, "'atoms_lst' should be provided if 'xyz_path' is None."
        else:
            assert atoms_lst is None, "'atoms_lst' should be None if 'xyz_path' is provided."
            atoms_lst = ase_io.read(xyz_path, index=":")
        self.atoms_lst = atoms_lst
        self.xyz_path = xyz_path
        self._node_count = None
        self._node_label_count = None

    def _len(self):
        return len(self.atoms_lst)

    def _get_item(self, inner_idx, transform: bool = True):
        atoms = self.atoms_lst[inner_idx]

        if transform and self.transform is not None:
            return self.transform(atoms)
        else:
            return atoms

    @property
    def node_count(self) -> dict:
        if self._node_count is None:
            counter = Counter(len(atoms) for atoms in self.atoms_lst)
            counter = dict(sorted(counter.items()))
            self._node_count = counter
        return self._node_count

    @property
    def node_label_count(self) -> dict:
        if self._node_label_count is None:
            counter = Counter(n for atoms in self.atoms_lst for n in atoms.get_atomic_numbers())
            counter = dict(sorted(counter.items()))
            self._node_label_count = counter
        return self._node_label_count

    @property
    def num_dimensions(self) -> int:
        return 3
