"""Augmentation strategy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


SUPPORTED_AUGMENTATION_KEYS = {
    "horizontal_flip",
    "vertical_flip",
    "random_rotate_90",
}

SUPPORTED_AUGMENTATION_STRATEGIES = {
    "none",
    "baseline",
    "custom",
}

BASELINE_AUGMENTATION_PROBS = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.5,
    "random_rotate_90": 0.5,
}


@dataclass
class AugmentationsConfig:
    """Config-driven augmentation strategy and probabilities."""

    train_strategy: str = "baseline"
    val_strategy: str = "none"
    test_strategy: str = "none"
    train: dict[str, float] = field(default_factory=lambda: dict(BASELINE_AUGMENTATION_PROBS))
    val: dict[str, float] = field(default_factory=dict)
    test: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        self._validate_strategy("augmentations.train_strategy", self.train_strategy)
        self._validate_strategy("augmentations.val_strategy", self.val_strategy)
        self._validate_strategy("augmentations.test_strategy", self.test_strategy)
        self._validate_probs("augmentations.train", self.train)
        self._validate_probs("augmentations.val", self.val)
        self._validate_probs("augmentations.test", self.test)

    @staticmethod
    def _validate_strategy(field_name: str, strategy: str) -> None:
        if strategy not in SUPPORTED_AUGMENTATION_STRATEGIES:
            options = ", ".join(sorted(SUPPORTED_AUGMENTATION_STRATEGIES))
            raise ValueError(f"{field_name} must be one of: {options}")

    @staticmethod
    def _validate_probs(field_name: str, probs: dict[str, float]) -> None:
        for key, value in probs.items():
            if key not in SUPPORTED_AUGMENTATION_KEYS:
                options = ", ".join(sorted(SUPPORTED_AUGMENTATION_KEYS))
                raise ValueError(f"{field_name} contains unsupported transform '{key}'. Allowed: {options}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name}.{key} must be in [0, 1], got {value}")
