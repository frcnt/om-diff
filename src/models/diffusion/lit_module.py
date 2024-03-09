import copy
import dataclasses
import itertools
import os
import time
import traceback
from typing import Optional, Union, Callable
from typing import Sequence

import torch

from src.data.base_datamodule import BaseDataModule
from src.data.components import Batch
from src.data.utils import convert_batch_to_atoms, save_images
from src.models.atomistic import AtomisticModel
from src.models.base_lit_module import BaseLitModule
from src.models.connectivity import Connectivity
from src.models.diffusion.loss import DiffusionL2Loss, DiffusionLoss
from src.models.diffusion.model import OMDiff
from src.models.diffusion.noise.model import MaskedNormalNoiseModel
from src.models.diffusion.sampling import BaseSampler
from src.models.diffusion.sampling.conditional import MaskedSampler


@dataclasses.dataclass
class DiffusionHParams:
    val_sample_every: int
    num_val_samples: int
    sampling_batch_size: int
    num_final_samples: int
    yield_final_samples_every: int
    max_sampling_try: int = 10
    scale_positions: float = 1.0


class DiffLitModule(BaseLitModule):
    om_diff: OMDiff

    def __init__(
            self,
            train_loss_module: DiffusionLoss,
            val_test_loss_module: DiffusionLoss,
            optimizer: torch.optim.Optimizer,
            diffusion_hp: DiffusionHParams,
            sampler: BaseSampler,
            scheduler: torch.optim.lr_scheduler = None,
            dm: BaseDataModule = None,
    ):
        super().__init__(optimizer=optimizer, scheduler=scheduler, dm=dm)
        self.train_loss_module = train_loss_module
        self.val_test_loss_module = val_test_loss_module
        self.sampler = sampler
        self.node_labels = dm.dataset.node_labels

    def forward(self, batch: Batch, ts: torch.Tensor):
        return self.om_diff(batch, ts)

    def step(self, batch, batch_idx, prefix: str):
        is_train = prefix.startswith("train")
        loss_module = self.train_loss_module if is_train else self.val_test_loss_module

        loss, loss_dict = loss_module(self.om_diff, batch)
        self.log(
            f"{prefix}/loss",
            loss,
            batch_size=batch.num_data,
            on_step=is_train,
            on_epoch=not is_train,
        )
        self.log_dict(
            {f"{prefix}/{k}": v for k, v in loss_dict.items()},
            batch_size=batch.num_data,
            on_step=is_train,
            on_epoch=not is_train,
        )
        return loss

    def on_validation_end(self) -> None:
        # sample from the model
        diffusion_hp: DiffusionHParams = self.hparams.diffusion_hp

        current_epoch = self.trainer.current_epoch
        log_dir = os.path.join(self.trainer.log_dir, "val_samples")
        os.makedirs(
            log_dir,
            exist_ok=True,
        )
        log_file = os.path.join(log_dir, f"{current_epoch:03d}.xyz")

        if (
                self.trainer.current_epoch + 1
        ) % diffusion_hp.val_sample_every == 0 and self.trainer.current_epoch > 0:
            start_time = time.time()
            print(f"Start validation sampling, epoch {current_epoch:03d}")
            self.sample(diffusion_hp, log_file=log_file)
            end_time = time.time()
            print(f"It took: {end_time - start_time}.")

    def on_test_end(self) -> None:
        """
        Saves a xyz with 'self.hparams.diffusion_hp.num_final_samples' denoising trajectories every
        'self.hparams.diffusion_hp.yield_final_samples_every' image saved.

        The trajectory of sample 'i' can be extracted by reading every 'self.hparams.diffusion_hp.num_final_samples'
        structure starting from index 'i'.

        """
        log_dir = os.path.join(self.trainer.log_dir, "test_samples")
        os.makedirs(
            log_dir,
            exist_ok=True,
        )
        log_file = os.path.join(log_dir, f"trajectories.xyz")
        hp: DiffusionHParams = self.hparams.diffusion_hp
        self.sample_trajectories(hp, log_file=log_file)


    def sample(
            self, hp: DiffusionHParams, log_file: str, save_intermediate: bool = False, **kwargs
    ):
        remaining_num_samples = hp.num_val_samples
        samples_atoms = []
        already_tried = 0
        while remaining_num_samples > 0 and already_tried < hp.max_sampling_try:
            print(f"{remaining_num_samples} remaining samples. Try: {already_tried}.")
            num_samples = min(remaining_num_samples, hp.sampling_batch_size)
            try:
                _, samples = next(
                    self.sampler.generate(
                        om_diff=self.om_diff,
                        num_samples=num_samples,
                        yield_every=-1,
                        **kwargs,
                    )
                )
                samples = samples.to("cpu")
                samples_atoms.extend(
                    convert_batch_to_atoms(
                        samples, node_labels=self.node_labels, scale_positions=hp.scale_positions
                    )
                )
                remaining_num_samples -= num_samples
                already_tried = 0  # reset counter
            except Exception as e:
                print(f"An exception was thrown during sampling, will try again ({e}).")
                traceback.print_tb(e.__traceback__)
                traceback.print_exc()
                already_tried += 1  # increase counter
            if save_intermediate:
                save_images(samples_atoms, filename=log_file)
        save_images(samples_atoms, filename=log_file)

    def sample_trajectories(self, hp: DiffusionHParams, log_file: str):
        remaining_num_samples = hp.num_final_samples
        samples_atoms: dict[int, list] = {}
        while remaining_num_samples > 0:
            num_samples = min(remaining_num_samples, hp.sampling_batch_size)
            remaining_num_samples -= num_samples
            for t, samples in self.sampler.generate(
                    om_diff=self.om_diff, num_samples=num_samples, yield_every=hp.yield_final_samples_every
            ):
                if t not in samples_atoms:
                    samples_atoms[t] = []
                samples_cpu = copy.deepcopy(samples).to(
                    "cpu"
                )  # copy needed as samples is used by the generator
                samples_atoms[t].extend(
                    convert_batch_to_atoms(
                        samples_cpu,
                        node_labels=self.node_labels,
                        scale_positions=hp.scale_positions,
                    )
                )

        # concatenate in order
        samples_atoms: Sequence = list(
            itertools.chain.from_iterable(samples_atoms[k] for k in sorted(samples_atoms))
        )
        save_images(samples_atoms, filename=log_file)


class OMDiffLitModule(DiffLitModule):
    def __init__(
            self,
            denoising_net: AtomisticModel,
            train_loss_module: DiffusionL2Loss,
            val_test_loss_module: DiffusionL2Loss,
            noise_model: MaskedNormalNoiseModel,
            optimizer: torch.optim.Optimizer,
            diffusion_hp: DiffusionHParams,
            sampler: MaskedSampler,
            connectivity_module: Optional[Union[Connectivity, Callable[[Batch], Batch]]] = None,
            scheduler: torch.optim.lr_scheduler = None,
            dm: BaseDataModule = None,
    ):
        super().__init__(
            train_loss_module=train_loss_module,
            val_test_loss_module=val_test_loss_module,
            diffusion_hp=diffusion_hp,
            optimizer=optimizer,
            sampler=sampler,
            scheduler=scheduler,
            dm=dm,
        )
        self.save_hyperparameters(
            ignore=[
                "denoising_net",
                "train_loss_module",
                "val_test_loss_module",
                "noise_model",
                "connectivity_module",
                "sampler",
                "dm",
            ]
        )

        self.om_diff: OMDiff = OMDiff.from_data(
            denoising_net=denoising_net,
            noise_model=noise_model,
            dataset=dm.dataset,
            connectivity_module=connectivity_module,
        )
        self.sampler = sampler

    def forward(self, batch: Batch, ts: torch.Tensor):
        return self.edm(batch, ts)
