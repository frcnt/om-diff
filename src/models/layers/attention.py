from typing import Optional

import torch
import torch.nn as nn

from src.data.components import Batch


class MultiHeadAttentionReadout(nn.Module):
    """
    Converts a set to a vector
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        num_heads: int,
        readout_net: nn.Module,
        input_key: str,
        splits_key: Optional[str] = None,
        output_key: Optional[str] = None,
    ):
        super().__init__()
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(input_dim, num_heads) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.readout_net = readout_net
        self.input_key = input_key
        self.output_key = input_key if output_key is None else output_key
        self.splits_key = splits_key
        self.model_outputs = [self.output_key]

    def forward(self, inputs: Batch) -> Batch:
        x = getattr(inputs, self.input_key)
        splits = getattr(inputs, self.splits_key)

        x = self.layer_norm(x)
        sequences = torch.split(x, splits.detach().cpu().tolist())
        masks = [
            torch.zeros(size=s.shape[0:1], dtype=torch.bool, device=x.device) for s in sequences
        ]
        masks = nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=True)

        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=False)
        query = torch.ones(
            size=(1, padded.shape[1], padded.shape[2]),
            device=padded.device,
            dtype=padded.dtype,
        )
        for att_layer in self.attention_layers:
            query, _ = att_layer(query, padded, padded, key_padding_mask=masks)

        x = self.readout_net(query.squeeze(0))

        setattr(inputs, self.output_key, x)
        return inputs
