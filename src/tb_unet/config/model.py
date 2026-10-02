"""Model architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """U-Net architecture settings."""

    in_channels: int | None = None
    num_classes: int = 3
    base_channels: int = 64
    depth: int = 4
    use_attention: bool = False
    bilinear: bool = False

    def validate(self) -> None:
        if self.in_channels is not None and self.in_channels < 1:
            raise ValueError("model.in_channels must be >= 1 when provided")
        if self.num_classes < 2:
            raise ValueError("model.num_classes must be >= 2")
        if self.base_channels < 1:
            raise ValueError("model.base_channels must be >= 1")
        if not 1 <= self.depth <= 6:
            raise ValueError("model.depth must be in [1, 6]")
