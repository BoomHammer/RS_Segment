import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.register_buffer(
            "denom",
            torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d),
            persistent=False,
        )

    def forward(self, batch_positions):
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table
