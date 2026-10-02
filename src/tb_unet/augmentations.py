"""Config-driven augmentation builders for TB U-Net."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tb_unet.config import (
    AugmentationsConfig,
    BASELINE_AUGMENTATION_PROBS,
    SUPPORTED_AUGMENTATION_KEYS,
)


def _as_config(config: AugmentationsConfig | Mapping[str, Any] | None) -> AugmentationsConfig:
    if config is None:
        cfg = AugmentationsConfig()
    elif isinstance(config, AugmentationsConfig):
        cfg = config
    else:
        cfg = AugmentationsConfig(**dict(config))
    cfg.validate()
    return cfg


def _strategy_probabilities(strategy: str, overrides: dict[str, float]) -> dict[str, float]:
    if strategy == "none":
        return {}
    if strategy == "baseline":
        probs = dict(BASELINE_AUGMENTATION_PROBS)
        probs.update(overrides)
        return probs
    if strategy == "custom":
        return dict(overrides)
    raise ValueError(f"Unsupported augmentation strategy: {strategy}")


def _build_compose(probabilities: dict[str, float]):
    if not probabilities:
        return None

    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "Albumentations is required when augmentations are enabled. "
            "Install it and re-run training."
        ) from exc

    builders = {
        "horizontal_flip": lambda p: A.HorizontalFlip(p=p),
        "vertical_flip": lambda p: A.VerticalFlip(p=p),
        "random_rotate_90": lambda p: A.RandomRotate90(p=p),
    }

    transforms = []
    for name, prob in probabilities.items():
        if name not in SUPPORTED_AUGMENTATION_KEYS:
            supported = ", ".join(sorted(SUPPORTED_AUGMENTATION_KEYS))
            raise ValueError(f"Unsupported transform '{name}'. Supported: {supported}")
        if prob <= 0:
            continue
        transforms.append(builders[name](prob))

    if not transforms:
        return None

    return A.Compose(transforms)


def get_split_probabilities(
    split: str,
    config: AugmentationsConfig | Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Resolve augmentation probabilities for a dataset split."""
    cfg = _as_config(config)

    if split == "train":
        return _strategy_probabilities(cfg.train_strategy, cfg.train)
    if split == "val":
        return _strategy_probabilities(cfg.val_strategy, cfg.val)
    if split == "test":
        return _strategy_probabilities(cfg.test_strategy, cfg.test)

    raise ValueError(f"Unsupported split '{split}'. Expected train/val/test")


def build_split_transform(
    split: str,
    config: AugmentationsConfig | Mapping[str, Any] | None = None,
):
    """Build albumentations transform for a split from config."""
    probabilities = get_split_probabilities(split, config)
    return _build_compose(probabilities)


def build_transforms(
    config: AugmentationsConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build train/val/test transforms from augmentation config."""
    return {
        "train": build_split_transform("train", config),
        "val": build_split_transform("val", config),
        "test": build_split_transform("test", config),
    }


__all__ = [
    "build_split_transform",
    "build_transforms",
    "get_split_probabilities",
]
