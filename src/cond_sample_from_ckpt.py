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
from src.data.components import Batch
from src.data.utils import save_json
from src.models.base_lit_module import BaseLitModule
from src.models.diffusion.lit_module import OMDiffLitModule, DiffusionHParams
from src.models.diffusion.sampling.conditional import RegressorGuidedMaskedSampler
from src.models.regression.lit_module import TimeConditionedRegressorLitModule

dotenv.load_dotenv(override=True)


def parse_cmd():
    parser = argparse.ArgumentParser(description="Script for conditional sampling from a checkpoint.")
    parser.add_argument(
        "--diffusion_dir_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--regressor_dir_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--guidance_strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--target_key",
        type=str,
        default="binding_energy",
    )
    parser.add_argument(
        "--target_value",
        type=float,
        default=-27.55,
    )
    parser.add_argument("--clamp_value", type=float, default=1.0)
    parser.add_argument("--feedback_from_step", type=int, default=1000)
    parser.add_argument("--mc_idx", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    return parser.parse_args()


def setup_paths(args: argparse.Namespace):
    samples_path = os.path.join(
        args.diffusion_dir_path,
        "conditional_samples",
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
    )
    os.makedirs(samples_path)
    save_json(vars(args), json_path=os.path.join(samples_path, "cmd_args.json"))

    # regressor path
    regressor_ckpt_dir_path = os.path.join(args.regressor_dir_path, "checkpoints")
    ckpt_paths = [
        os.path.join(regressor_ckpt_dir_path, path)
        for path in sorted(os.listdir(regressor_ckpt_dir_path))
        if path != "last.ckpt"
    ]
    assert len(ckpt_paths) > 0
    regressor_ckpt = ckpt_paths[-1]

    dict_paths = {
        "diffusion_config": os.path.join(args.diffusion_dir_path, ".hydra", "config.yaml"),
        "diffusion_ckpt": os.path.join(args.diffusion_dir_path, "checkpoints", "last.ckpt"),
        "regressor_config": os.path.join(args.regressor_dir_path, ".hydra", "config.yaml"),
        "regressor_ckpt": regressor_ckpt,
        "samples": samples_path,
    }
    return dict_paths


def load_diffusion_model(
        paths: dict, args: argparse.Namespace, regressor: TimeConditionedRegressorLitModule
):
    print("Loading diffusion model...")
    device = torch.device(args.device)

    diffusion_cfg = OmegaConf.load(paths["diffusion_config"])

    print(f"Read data from checkpoint.")
    print(f"> Instantiating datamodule <{diffusion_cfg.data._target_}>")
    diffusion_datamodule: BaseDataModule = hydra.utils.instantiate(diffusion_cfg.data)
    diffusion_datamodule.setup(stage="test")

    print(f"> Instantiating model <{diffusion_cfg.model._target_}>")
    diffusion_model: OMDiffLitModule = hydra.utils.instantiate(
        diffusion_cfg.model, dm=diffusion_datamodule
    )

    diffusion_ckpt = torch.load(paths["diffusion_ckpt"], map_location=device)

    (missing_keys, unexpected_keys) = diffusion_model.load_state_dict(
        diffusion_ckpt["state_dict"], strict=False
    )
    diffusion_model = diffusion_model.to(device)

    if len(missing_keys) > 0:
        warnings.warn(
            f"Some keys were missing from the 'state_dict' ({missing_keys}), this might lead to unexpected results."
        )

    if len(unexpected_keys) > 0:
        warnings.warn(
            f"Some keys were unexpected in 'state_dict' ({unexpected_keys}), this might lead to unexpected results."
        )

    def target_function(inputs: Batch, ts: torch.Tensor):
        preds = regressor.forward(inputs, ts)
        energy = -torch.sum(torch.square(preds[args.target_key] - args.target_value))
        return energy

    sampler_kwargs = dict(
        guidance_strength=args.guidance_strength,
        target_function=target_function,
        clamp_value=args.clamp_value,
        feedback_from_step=args.feedback_from_step,
    )

    sampler_kwargs["scale"] = diffusion_model.sampler.scale
    diffusion_model.sampler = RegressorGuidedMaskedSampler(**sampler_kwargs)

    print(f"Loaded sampler: '{type(diffusion_model.sampler)}'.")

    # create datastructure and directory where to store samples
    diffusion_hp: DiffusionHParams = diffusion_model.hparams.diffusion_hp
    diffusion_hp.num_val_samples = args.n

    return diffusion_model, diffusion_hp


def load_regressor(paths: dict, args):
    print("Loading regressor...")

    cfg = OmegaConf.load(paths["regressor_config"])

    device = torch.device(args.device)

    print(f"Read data from checkpoint.")
    print(f"> Instantiating datamodule <{cfg.data._target_}>")
    datamodule: BaseDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")

    print(f"> Instantiating model <{cfg.model._target_}>")
    model: BaseLitModule = hydra.utils.instantiate(cfg.model, dm=datamodule)
    model = model.to(device)

    print("> Loading checkpoint from ", paths["regressor_ckpt"])
    ckpt = torch.load(paths["regressor_ckpt"], map_location=device)

    (missing_keys, unexpected_keys) = model.load_state_dict(ckpt["state_dict"], strict=False)
    if len(missing_keys) > 0:
        warnings.warn(
            f"Some keys were missing from the 'state_dict' ({missing_keys}), this might lead to unexpected results."
        )

    if len(unexpected_keys) > 0:
        warnings.warn(
            f"Some keys were unexpected in 'state_dict' ({unexpected_keys}), this might lead to unexpected results."
        )

    return model


if __name__ == "__main__":
    L.seed_everything(42)

    cmd_args = parse_cmd()
    device = torch.device(cmd_args.device)

    print(f"CUDA device available: {torch.cuda.is_available()} - Chosen device: {device}")
    paths = setup_paths(cmd_args)
    print(f"Set up paths: {paths}")

    regressor = load_regressor(paths=paths, args=cmd_args)
    diffusion_model, diffusion_hp = load_diffusion_model(
        paths=paths, args=cmd_args, regressor=regressor
    )

    samples_path = os.path.join(paths["samples"], "samples.xyz")

    sample_kwargs = dict(
        hp=diffusion_hp,
        log_file=os.path.join(paths["samples"], "samples.xyz"),
        save_intermediate=True,
    )

    sample_kwargs["zs_idx"] = torch.tensor(
        [cmd_args.mc_idx],
        device=cmd_args.device,
    )

    start_time = time.time()
    print(f"Start sampling with config ({diffusion_hp}). This might take some time...")
    # with torch.inference_mode():
    diffusion_model.sample(**sample_kwargs)
    end_time = time.time()
    print(
        f"Done Sampling. Samples are available at {samples_path}. It took: {end_time - start_time}."
    )
