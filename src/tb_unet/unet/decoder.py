"""U-Net decoder module."""

from __future__ import annotations

import torch
import torch.nn as nn

from tb_unet.unet.attention import AttentionGate
from tb_unet.unet.blocks import UpBlock


class UNetDecoder(nn.Module):
    """Standalone U-Net decoder consuming bottleneck + skip list."""

    def __init__(
        self,
        skip_channels: list[int],
        bottleneck_channels: int,
        num_classes: int = 3,
        bilinear: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()
        if not skip_channels:
            raise ValueError("skip_channels must not be empty")

        self.skip_channels = list(skip_channels)
        self.bottleneck_channels = bottleneck_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.use_attention = use_attention

        self.up_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        current_channels = bottleneck_channels
        for skip_ch in reversed(self.skip_channels):
            self.up_blocks.append(
                UpBlock(
                    decoder_channels=current_channels,
                    skip_channels=skip_ch,
                    out_channels=skip_ch,
                    bilinear=bilinear,
                )
            )

            if use_attention:
                inter_channels = max(skip_ch // 2, 1)
                self.attention_gates.append(
                    AttentionGate(
                        gate_channels=current_channels,
                        skip_channels=skip_ch,
                        inter_channels=inter_channels,
                    )
                )

            current_channels = skip_ch

        self.outc = nn.Conv2d(current_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        if len(skips) != len(self.up_blocks):
            raise ValueError(
                f"Expected {len(self.up_blocks)} skip tensors, received {len(skips)}"
            )

        for idx, up in enumerate(self.up_blocks):
            skip = skips[-(idx + 1)]
            if self.use_attention:
                skip = self.attention_gates[idx](skip, x)
            x = up(x, skip)

        return self.outc(x)
