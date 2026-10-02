"""Loss-related configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LossType = Literal["combined", "focal_dice"]


@dataclass
class LossConfig:
    """Loss weighting and class-weight settings."""

    loss_type: LossType = "combined"
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_gamma: float = 2.0
    class_weights: list[float] | None = None
    class_weight_method: str = "inverse_sqrt_freq"

    def validate(self) -> None:
        if self.loss_type not in {"combined", "focal_dice"}:
            raise ValueError("loss.loss_type must be one of: combined, focal_dice")
        if self.ce_weight < 0:
            raise ValueError("loss.ce_weight must be >= 0")
        if self.dice_weight < 0:
            raise ValueError("loss.dice_weight must be >= 0")
        if self.focal_gamma <= 0:
            raise ValueError("loss.focal_gamma must be > 0")
        if self.class_weights is not None:
            if not self.class_weights:
                raise ValueError("loss.class_weights must be null or a non-empty list")
            if any(w <= 0 for w in self.class_weights):
                raise ValueError("loss.class_weights values must be > 0")
