"""
Greatly inspired from SchNetPack's implementation.
"""
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn

from src.data.components import Batch

__all__ = ["AtomisticModel"]


class AtomisticModel(nn.Module):
    """
    Wrapper for all models.
    Includes:
        - input modules
        - backbone
        - output modules
    The forward processes a batch and return a dict with the specified outputs.
    The outputs can be provided as arguments and/or will be collected from inner modules.
    """

    def __init__(
            self,
            backbone: nn.Module,
            input_modules: Optional[OrderedDict[str, nn.Module]] = None,
            output_modules: Optional[OrderedDict[str, nn.Module]] = None,
            additional_model_outputs: Optional[list[str]] = None,
    ):
        super(AtomisticModel, self).__init__()
        self.backbone = backbone
        self.input_modules = nn.ModuleDict(input_modules)
        self.output_modules = nn.ModuleDict(output_modules)

        self.model_outputs = self.collect_outputs(additional_model_outputs)
        self.required_derivatives = self.collect_derivatives()

    def collect_outputs(self, additional_model_outputs: Optional[list[str]]) -> list[str]:
        """
        Collects all the keys that need to be extracted at the end of the forward pass.
        """
        if additional_model_outputs is None:
            model_outputs = set()
        else:
            model_outputs = set(additional_model_outputs)
        for m in self.modules():
            if hasattr(m, "model_outputs") and m.model_outputs is not None:
                model_outputs.update(m.model_outputs)
        model_outputs = list(model_outputs)

        return model_outputs

    def collect_derivatives(self) -> list[str]:
        """
        Collects all the keys wrt. which the derivative must be computed.
        """
        self.required_derivatives = None
        required_derivatives = set()
        for m in self.modules():
            if hasattr(m, "required_derivatives") and m.required_derivatives is not None:
                required_derivatives.update(m.required_derivatives)
        required_derivatives = list(required_derivatives)

        return required_derivatives

    def initialize_derivatives(self, inputs: Batch) -> Batch:
        for p in self.required_derivatives:
            if hasattr(inputs, p):
                x = getattr(inputs, p)
                x.requires_grad_()

        return inputs

    def extract_outputs(self, inputs: Batch) -> dict[str, torch.Tensor]:
        results = {k: getattr(inputs, k) for k in self.model_outputs}
        return results

    def forward(self, inputs: Batch) -> dict[str, torch.Tensor]:
        inputs = self.initialize_derivatives(inputs)

        # apply all input modules
        for m in self.input_modules:
            inputs = self.input_modules[m](inputs)

        inputs = self.backbone(inputs)

        for m in self.output_modules:
            inputs = self.output_modules[m](inputs)

        out = self.extract_outputs(inputs)

        return out
