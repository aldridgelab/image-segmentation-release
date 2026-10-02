"""Core U-Net convolution/downsampling/upsampling blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Encoder block: max-pool followed by ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """
    Decoder block: upsample current feature map, concatenate skip, then ConvBlock.

    `decoder_channels` is the current decoder channel count.
    `skip_channels` is the channel count of the corresponding skip connection.
    """

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int | None = None,
        bilinear: bool = False,
    ):
        super().__init__()
        out_channels = out_channels if out_channels is not None else skip_channels
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(decoder_channels, skip_channels, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(decoder_channels, skip_channels, kernel_size=2, stride=2)

        self.conv = ConvBlock(skip_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle odd spatial shapes.
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            x = F.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
