from abc import abstractmethod
from typing import Optional

import ase
import torch
from ase.db.row import AtomsRow

from src.data.components import AtomsData
from src.data.components.transforms.base import Transform


class AtomsRowTransform(Transform):
    @abstractmethod
    def __call__(self, atoms_row: AtomsRow) -> AtomsData:
        raise NotImplementedError()


class AtomsRowToAtomsDataTransform(AtomsRowTransform):
    """
    Transforms an ASE DB AtomsRow object to an AtomsData object
    and extracts specified properties from it.
    """

    def __init__(self, extract_properties: Optional[list[str]] = None):
        self.extract_properties = [] if extract_properties is None else extract_properties

    def __call__(self, atoms_row: AtomsRow) -> AtomsData:
        atoms = atoms_row.toatoms()
        pty_dict = {}
        for pty_name in self.extract_properties:
            if hasattr(atoms_row, pty_name):
                pty = getattr(atoms_row, pty_name)
            elif hasattr(atoms_row, "data") and pty_name in atoms_row.data:
                pty = atoms_row.data[pty_name]
            else:
                raise AttributeError(f"'AtomsRow' has no property: {pty_name}")

            # FIXME: get rid of the squeeze() all over the place
            pty = torch.atleast_1d(torch.tensor(pty, dtype=torch.get_default_dtype()).squeeze())
            pty_dict[pty_name] = pty
        # create AtomsData object
        atoms_data = AtomsData(
            node_features=torch.tensor(atoms.get_atomic_numbers()).unsqueeze(1),
            node_positions=torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype()),
            **pty_dict,
        )
        return atoms_data


class AtomsToAtomsDataTransform(Transform):
    """
    Transforms an ASE Atoms object to an AtomsData object
    and extracts specified properties from it.
    """

    def __call__(self, atoms: ase.Atoms) -> AtomsData:
        atoms_data = AtomsData(
            node_features=torch.tensor(atoms.get_atomic_numbers()).unsqueeze(1),
            node_positions=torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype()),
        )
        return atoms_data
