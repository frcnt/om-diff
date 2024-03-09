import argparse
import os
import time
import warnings
from datetime import datetime

import dotenv
import hydra
import lightning as L
import torch
from omegaconf import OmegaConf

from src.data.base_datamodule import BaseDataModule
from src.data.utils import save_json
from src.models.diffusion.lit_module import OMDiffLitModule, DiffusionHParams

dotenv.load_dotenv(override=True)


def parse_cmd():
    parser = argparse.ArgumentParser(description="Script for sampling from a checkpoint.")
    parser.add_argument(
        "--dir_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument("--use_last_ckpt", action="store_true")

    return parser.parse_args()


def setup_paths(args: argparse.Namespace):
    samples_path = os.path.join(
        args.dir_path, "samples", datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    )
    os.makedirs(samples_path)

    ckpt_dir_path = os.path.join(args.dir_path, "checkpoints")
    if args.use_last_ckpt:
        ckpt_path = os.path.join(ckpt_dir_path, "last.ckpt")
    else:
        ckpt_paths = [
            os.path.join(ckpt_dir_path, path)
            for path in sorted(os.listdir(ckpt_dir_path))
            if path != "last.ckpt"
        ]
        assert len(ckpt_paths) > 0
        ckpt_path = ckpt_paths[-1]

    args_dict = {**vars(args), "ckpt_path": ckpt_path}
    save_json(args_dict, json_path=os.path.join(samples_path, "cmd_args.json"))

    dict_paths = {
        "config": os.path.join(args.dir_path, ".hydra", "config.yaml"),
        "ckpt": ckpt_path,
        "samples": samples_path,
    }
    return dict_paths


if __name__ == "__main__":
    L.seed_everything(42)

    cmd_args = parse_cmd()
    device = torch.device(cmd_args.device)

    print(f"CUDA device available: {torch.cuda.is_available()} - Chosen device: {device}")
    paths = setup_paths(cmd_args)
    print(f"Set up paths: {paths}")
    cfg = OmegaConf.load(paths["config"])

    print(f"Read data from checkpoint.")
    print(f"> Instantiating datamodule <{cfg.data._target_}>")
    datamodule: BaseDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")

    print(f"> Instantiating model <{cfg.model._target_}>")
    model: OMDiffLitModule = hydra.utils.instantiate(cfg.model, dm=datamodule)

    ckpt = torch.load(paths["ckpt"], map_location=device)

    (missing_keys, unexpected_keys) = model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device)

    if len(missing_keys) > 0:
        warnings.warn(
            f"Some keys were missing from the 'state_dict' ({missing_keys}), this might lead to unexpected results."
        )

    if len(unexpected_keys) > 0:
        warnings.warn(
            f"Some keys were unexpected in 'state_dict' ({unexpected_keys}), this might lead to unexpected results."
        )

    # create datastructure and directory where to store samples
    diffusion_hp: DiffusionHParams = model.hparams.diffusion_hp
    diffusion_hp.num_val_samples = cmd_args.n

    samples_path = os.path.join(paths["samples"], "samples.xyz")
    start_time = time.time()
    print(f"Start sampling with config ({diffusion_hp}). This might take some time...")
    with torch.inference_mode():
        model.sample(
            hp=diffusion_hp,
            log_file=os.path.join(paths["samples"], "samples.xyz"),
            save_intermediate=True,
        )
    end_time = time.time()
    print(
        f"Done Sampling. Samples are available at {samples_path}. It took: {end_time - start_time}."
    )
