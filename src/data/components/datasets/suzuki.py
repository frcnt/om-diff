import os
import shutil
import urllib.request as requests

import ase
import ase.io
import numpy as np
import pandas as pd
import tqdm
from ase.db import connect
from ase.units import kcal, mol

from src.data.components.datasets.ase_dataset import ASEDBDataset
from src.data.components.transforms.base import Transform

__all__ = ["SuzukiDataset", "Suzuki4Dataset", "SuzukiFFDataset"]


class BaseSuzukiDataset(ASEDBDataset):
    """
    Suzuki Dataset from doi:10.1039/C8SC01949E.
    """

    distant_url = "https://archive.materialscloud.org/record/file?filename={}&record_id=56"
    filenames = {
        "structures.tar.gz": "md5:030cd6a0e4fc77b0974e9ceb33fe8ce8",
        "optimized.tar.gz": "md5:5e6865d8715bd983a2b814d550108ba5",
        "energies.tar.gz": "md5:275389a88c051b4e41b84b6c71c298e6",
    }
    properties = {
        "binding_energy": kcal / mol,
    }
    expected_length = 7054

    energies_path: str = "energies/CompBindEn.txt"
    structures_path: str = ...

    def __init__(
        self,
        db_path,
        transform: Transform = lambda _x: _x,
        download: bool = False,
        idx_subset: np.ndarray = None,
        in_memory: bool = True,
        **kwargs,
    ):
        if download:
            dir_path = os.path.dirname(db_path)
            db_name = os.path.basename(db_path)
            self.download_dataset(dir_path, db_name)
        super(BaseSuzukiDataset, self).__init__(
            db_path=db_path,
            transform=transform,
            idx_subset=idx_subset,
            in_memory=in_memory,
            **kwargs,
        )

    @classmethod
    def download_dataset(
        cls, dir_path, db_name: str = "suzuki.db", force_download: bool = True, clean: bool = True
    ):
        db_path = os.path.join(dir_path, db_name)
        energies_path = os.path.join(dir_path, cls.energies_path)
        structures_path = os.path.join(dir_path, cls.structures_path)

        archive_names = ["energies.tar.gz", "optimized.tar.gz"]

        for archive_name in archive_names:
            archive_path = os.path.join(dir_path, archive_name)
            if not os.path.exists(archive_path) or force_download:
                with tqdm.tqdm(
                    unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=archive_path
                ) as t:
                    requests.urlretrieve(
                        cls.distant_url.format(archive_name), filename=archive_path
                    )

                shutil.unpack_archive(archive_path, extract_dir=dir_path)

                if clean:
                    os.remove(archive_path)

        df = pd.read_csv(
            energies_path,
            sep=" ",
            skiprows=1,
            names=["name", "energy_2", "energy_4", "binding_energy_Ha", "binding_energy"],
            index_col=0,
        )

        with connect(db_path, append=False) as db:
            for structure_path in tqdm.tqdm(os.listdir(structures_path)):
                name = structure_path[:-6]
                kv_pairs = df.loc[name, :].to_dict()
                atoms = ase.io.read(os.path.join(structures_path, structure_path))

                db.write(atoms, key_value_pairs=kv_pairs)

        if clean:
            shutil.rmtree(os.path.join(dir_path, "energies"))
            shutil.rmtree(os.path.join(dir_path, "optimized"))

        print(f"Done downloading Suzuki, now located at {db_path}.")


class SuzukiDataset(BaseSuzukiDataset):
    structures_path = "optimized/DFTgeom2Lig"


class Suzuki4Dataset(BaseSuzukiDataset):
    structures_path = "optimized/DFTgeom4Lig"


class SuzukiFFDataset(ASEDBDataset):
    """
    Suzuki Dataset from doi:10.1039/C8SC01949E.
    """

    distant_url = "https://archive.materialscloud.org/record/file?filename={}&record_id=56"
    filenames = {
        "structures.tar.gz": "md5:030cd6a0e4fc77b0974e9ceb33fe8ce8",
    }
    expected_length = 25_116

    def __init__(
        self,
        db_path,
        transform: Transform = lambda _x: _x,
        download: bool = False,
        idx_subset: np.ndarray = None,
        in_memory: bool = True,
        **kwargs,
    ):
        if download:
            dir_path = os.path.dirname(db_path)
            db_name = os.path.basename(db_path)
            self.download_dataset(dir_path, db_name)
        super(SuzukiFFDataset, self).__init__(
            db_path=db_path,
            transform=transform,
            idx_subset=idx_subset,
            in_memory=in_memory,
            **kwargs,
        )

    @classmethod
    def download_dataset(
        cls,
        dir_path,
        db_name: str = "suzuki_ff.db",
        force_download: bool = True,
        clean: bool = True,
    ):
        db_path = os.path.join(dir_path, db_name)
        structures_path = os.path.join(dir_path, "structures/All2Lig")

        archive_names = [
            "structures.tar.gz",
        ]

        for archive_name in archive_names:
            archive_path = os.path.join(dir_path, archive_name)
            if not os.path.exists(archive_path) or force_download:
                with tqdm.tqdm(
                    unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=archive_path
                ) as t:
                    requests.urlretrieve(
                        cls.distant_url.format(archive_name), filename=archive_path
                    )

                shutil.unpack_archive(archive_path, extract_dir=dir_path)

                if clean:
                    os.remove(archive_path)

        with connect(db_path, append=False) as db:
            for structure_path in tqdm.tqdm(os.listdir(structures_path)):
                atoms = ase.io.read(os.path.join(structures_path, structure_path))

                db.write(atoms)

        if clean:
            shutil.rmtree(os.path.join(dir_path, "structures"))

        print(f"Done downloading Suzuki FF, now located at {db_path}.")


if __name__ == "__main__":
    from src.data.components.transforms.ase import AtomsRowToAtomsDataTransform

    dataset = Suzuki4Dataset(
        db_path="/Users/frjc/phd/projects/diffusion4md/data/suzuki-4.db",
        download=False,
        transform=AtomsRowToAtomsDataTransform(extract_properties=["binding_energy"]),
    )
    assert len(dataset) == dataset.expected_length
    print(dataset.node_label_count)
    print(list(dataset.node_label_count.keys()))
    print(dataset[0].__dict__)
