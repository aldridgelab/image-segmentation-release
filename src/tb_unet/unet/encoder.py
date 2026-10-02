"""U-Net encoder module."""

from __future__ import annotations

import torch
import torch.nn as nn

from tb_unet.unet.blocks import ConvBlock, DownBlock


class UNetEncoder(nn.Module):
    """Standalone U-Net encoder returning bottleneck and skip features."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth

        self.stem = ConvBlock(in_channels, base_channels)
        self.down_blocks = nn.ModuleList()

        channels = base_channels
        self.skip_channels = [base_channels]
        for level in range(depth):
            out_channels = channels * 2
            self.down_blocks.append(DownBlock(channels, out_channels))
            channels = out_channels
            if level < depth - 1:
                self.skip_channels.append(channels)

        self.bottleneck_channels = channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.stem(x)
        skips = [x]

        for idx, down in enumerate(self.down_blocks):
            x = down(x)
            if idx < len(self.down_blocks) - 1:
                skips.append(x)

        return x, skips
