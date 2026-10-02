"""Composable U-Net model definitions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tb_unet.unet.decoder import UNetDecoder
from tb_unet.unet.encoder import UNetEncoder


class UNet(nn.Module):
    """Standard U-Net composed from an encoder and decoder."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = False,
        use_attention: bool = False,
        encoder: UNetEncoder | None = None,
        decoder: UNetDecoder | None = None,
    ):
        super().__init__()

        self.encoder = encoder or UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
        )
        self.decoder = decoder or UNetDecoder(
            skip_channels=self.encoder.skip_channels,
            bottleneck_channels=self.encoder.bottleneck_channels,
            num_classes=num_classes,
            bilinear=bilinear,
            use_attention=use_attention,
        )

        self.in_channels = self.encoder.in_channels
        self.base_channels = self.encoder.base_channels
        self.depth = self.encoder.depth
        self.num_classes = self.decoder.num_classes
        self.bilinear = bilinear
        self.use_attention = use_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encoder(x)
        return self.decoder(bottleneck, skips)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class UNetWithAttention(UNet):
    """Compatibility wrapper for attention-enabled U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
            bilinear=bilinear,
            use_attention=True,
        )
