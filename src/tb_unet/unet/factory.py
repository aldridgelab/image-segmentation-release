"""Factory helpers for constructing U-Net models from config objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from tb_unet.unet.unet import UNet


def _get_value(config: object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def build_unet(config: object) -> nn.Module:
    """Build a U-Net from a model config-like object or mapping."""
    in_channels = _get_value(config, "in_channels")
    if in_channels is None:
        raise ValueError("model.in_channels must be set before calling build_unet")

    return UNet(
        in_channels=in_channels,
        num_classes=_get_value(config, "num_classes", 3),
        base_channels=_get_value(config, "base_channels", 64),
        depth=_get_value(config, "depth", 4),
        bilinear=_get_value(config, "bilinear", False),
        use_attention=_get_value(config, "use_attention", False),
    )
