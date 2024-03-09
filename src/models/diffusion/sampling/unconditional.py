from typing import Tuple, Iterator

import torch

from src.data.components import Batch
from src.models.diffusion.model import OMDiff
from src.models.diffusion.sampling import BaseSampler


class UnconditionalSampler(BaseSampler):
    @torch.no_grad()
    def generate(
        self,
        om_diff: OMDiff,
        num_samples: int = 1,
        num_nodes: torch.Tensor = None,
        yield_init: bool = False,
        yield_every: int = 1,
    ) -> Iterator[Tuple[int, Batch]]:  # return type generator
        return super(UnconditionalSampler, self).generate(
            om_diff=om_diff,
            num_samples=num_samples,
            num_nodes=num_nodes,
            yield_init=yield_init,
            yield_every=yield_every,
            condition_dict=None,
        )
