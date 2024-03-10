import argparse
import os
import warnings

import dotenv
import hydra
import lightning as L
import torch
import tqdm
from omegaconf import OmegaConf

from src.data.base_datamodule import BaseDataModule, get_dataloader
from src.data.components.datasets.ase_dataset import XYZDataset
from src.data.components.transforms.ase import AtomsToAtomsDataTransform
from src.data.components.transforms.base import Compose
from src.models.diffusion.lit_module import BaseLitModule

dotenv.load_dotenv(override=True)


def parse_cmd():
    parser = argparse.ArgumentParser(description="Script for predicting properties from a checkpoint.")
    parser.add_argument(
        "--predictor_dir_path", type=str, required=True, help="Path to the model's directory."
    )
    parser.add_argument(
        "--samples_path", type=str, required=True, help="Path to the samples' directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
    )
    parser.add_argument("--file_suffix", type=str, default=None)

    return parser.parse_args()


def setup_paths(args: argparse.Namespace):
    ckpt_dir_path = os.path.join(args.predictor_dir_path, "checkpoints")
    ckpt_paths = [
        os.path.join(ckpt_dir_path, path)
        for path in sorted(os.listdir(ckpt_dir_path))
        if path != "last.ckpt"
    ]
    assert len(ckpt_paths) > 0
    ckpt_path = ckpt_paths[-1]

    return {
        "config": os.path.join(args.predictor_dir_path, ".hydra", "config.yaml"),
        "ckpt": ckpt_path,
        "xyz": os.path.join(
            args.samples_path,
            "samples.xyz",
        ),
        "predictions": os.path.join(
            args.samples_path,
            "predictions.pt"
            if args.file_suffix is None
            else f"predictions_{cmd_args.file_suffix}.pt",
        ),
    }


if __name__ == "__main__":
    L.seed_everything(42)

    cmd_args = parse_cmd()

    paths = setup_paths(cmd_args)

    cfg = OmegaConf.load(paths["config"])

    device = torch.device(cmd_args.device)

    print(f"Read data from checkpoint.")
    print(f"> Instantiating datamodule <{cfg.data._target_}>")
    datamodule: BaseDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")

    print(f"> Instantiating model <{cfg.model._target_}>")
    model: BaseLitModule = hydra.utils.instantiate(cfg.model, dm=datamodule)
    model = model.to(device)

    print("> Loading checkpoint from ", paths["ckpt"])
    ckpt = torch.load(paths["ckpt"], map_location=device)

    (missing_keys, unexpected_keys) = model.load_state_dict(ckpt["state_dict"], strict=False)
    if len(missing_keys) > 0:
        warnings.warn(
            f"Some keys were missing from the 'state_dict' ({missing_keys}), this might lead to unexpected results."
        )

    if len(unexpected_keys) > 0:
        warnings.warn(
            f"Some keys were unexpected in 'state_dict' ({unexpected_keys}), this might lead to unexpected results."
        )

    # build dataloader with samples
    transform: Compose = datamodule.dataset.transform
    dm_hp = datamodule.hparams

    transform.transforms["row_to_atoms"] = AtomsToAtomsDataTransform()  # now reading from xyz

    predict_dataset = XYZDataset(xyz_path=paths["xyz"], transform=transform)
    predict_loader = get_dataloader(dataset=predict_dataset, shuffle=False, hp=dm_hp)

    outs = []
    with torch.inference_mode():
        for batch in tqdm.tqdm(predict_loader):
            batch = model.transfer_batch_to_device(batch, device=device, dataloader_idx=0)
            out = model(batch)
            outs.append(out)

    outs = {key: torch.cat([o[key].cpu() for o in outs], dim=0) for key in outs[0]}
    torch.save(outs, f=paths["predictions"])
