import json
from typing import Sequence, Optional, Union, Iterable

import ase.io
import torch
from ase import Atoms

from src.data.components import Batch


def convert_batch_to_atoms(
    batch: Batch,
    node_labels: Optional[Union[list[int], torch.Tensor]] = None,
    scale_positions: Optional[float] = 1.0,
) -> list[Atoms]:
    sections = batch.num_nodes.tolist()

    numbers = torch.argmax(batch.node_features, dim=-1)

    numbers = torch.split(numbers, split_size_or_sections=sections)
    positions = torch.split(
        batch.node_positions * scale_positions, split_size_or_sections=sections
    )

    if node_labels is not None:  # decode if node_labels is provided
        decoder = {i: v for (i, v) in enumerate(node_labels)}
        numbers = [[decoder[i.item()] for i in n] for n in numbers]

    return [Atoms(positions=r, numbers=z) for z, r in zip(numbers, positions)]


def save_images(images: Sequence[Atoms], filename: str):
    ase.io.write(filename=filename, images=images, format="xyz")


def save_json(json_dict: dict, json_path: str):
    with open(json_path, encoding="utf-8", mode="w") as fp:
        json.dump(json_dict, fp)


def read_json(json_path: str):
    with open(json_path, encoding="utf-8", mode="r") as fp:
        return json.load(fp)


def save_txt(line_itr: Iterable, txt_path: str):
    with open(txt_path, encoding="utf-8", mode="w") as fp:
        for line in line_itr:
            fp.write(f"{line}\n")
