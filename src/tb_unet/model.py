"""Backward-compatible model API shim for TB U-Net."""

from __future__ import annotations

import torch.nn as nn

from tb_unet.unet import (
    AttentionGate,
    ConvBlock,
    DownBlock,
    UNet,
    UNetDecoder,
    UNetEncoder,
    UNetWithAttention,
    UpBlock,
    build_unet,
)


def create_model(
    in_channels: int = 1,
    num_classes: int = 3,
    base_channels: int = 64,
    depth: int = 4,
    use_attention: bool = False,
    bilinear: bool = False,
) -> nn.Module:
    """Factory compatible with the legacy model API."""
    if use_attention:
        return UNetWithAttention(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
            bilinear=bilinear,
        )
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        depth=depth,
        bilinear=bilinear,
        use_attention=False,
    )


__all__ = [
    "AttentionGate",
    "ConvBlock",
    "DownBlock",
    "UpBlock",
    "UNetEncoder",
    "UNetDecoder",
    "UNet",
    "UNetWithAttention",
    "build_unet",
    "create_model",
]
