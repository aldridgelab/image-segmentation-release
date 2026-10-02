"""Top-level nested training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from tb_unet.config.augmentations import AugmentationsConfig
from tb_unet.config.data import DataConfig
from tb_unet.config.logging import LoggingConfig
from tb_unet.config.loss import LossConfig
from tb_unet.config.model import ModelConfig
from tb_unet.config.output import OutputConfig
from tb_unet.config.training import TrainingConfig

NESTED_SECTIONS = {
    "data",
    "model",
    "training",
    "loss",
    "output",
    "logging",
    "augmentations",
}

FLAT_KEYS = {
    "data_dir",
    "image_dir",
    "channels",
    "pad_multiple",
    "in_channels",
    "num_classes",
    "base_channels",
    "depth",
    "use_attention",
    "bilinear",
    "batch_size",
    "num_workers",
    "epochs",
    "lr",
    "weight_decay",
    "scheduler",
    "scheduler_patience",
    "scheduler_factor",
    "early_stopping_patience",
    "mixed_precision",
    "gradient_clip",
    "loss_type",
    "ce_weight",
    "dice_weight",
    "focal_gamma",
    "class_weights",
    "class_weight_method",
    "output_dir",
    "experiment_name",
    "save_every",
    "log_every",
    "wandb_enabled",
    "wandb_project",
    "wandb_entity",
    "wandb_tags",
}

FLAT_CONFIG_MIGRATION_MESSAGE = (
    "Looks like a pre-refactor flat config. "
    "Run `python tb_unet/scripts/migrate_config.py --input <old.yaml> --output <new.yaml>` to convert."
)


@dataclass
class TrainConfig:
    """Nested training configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    augmentations: AugmentationsConfig = field(default_factory=AugmentationsConfig)
    run_id: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        if not isinstance(data, Mapping):
            raise TypeError("config must be a mapping")

        if _looks_like_flat_config(data):
            raise ValueError(FLAT_CONFIG_MIGRATION_MESSAGE)

        cfg = cls(
            data=_build_dataclass(DataConfig, data.get("data", {})),
            model=_build_dataclass(ModelConfig, data.get("model", {})),
            training=_build_dataclass(TrainingConfig, data.get("training", {})),
            loss=_build_dataclass(LossConfig, data.get("loss", {})),
            output=_build_dataclass(OutputConfig, data.get("output", {})),
            logging=_build_dataclass(LoggingConfig, data.get("logging", {})),
            augmentations=_build_dataclass(AugmentationsConfig, data.get("augmentations", {})),
            run_id=data.get("run_id"),
        )
        cfg.validate()
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.loss.validate()
        self.output.validate()
        self.logging.validate()
        self.augmentations.validate()

        inferred_channels = len(self.data.channels)
        if self.model.in_channels is None:
            self.model.in_channels = inferred_channels

        if self.model.in_channels != inferred_channels:
            raise ValueError(
                "model.in_channels must match len(data.channels). "
                f"Got model.in_channels={self.model.in_channels}, len(data.channels)={inferred_channels}."
            )

        if self.loss.class_weights is not None and len(self.loss.class_weights) != self.model.num_classes:
            raise ValueError(
                "loss.class_weights length must equal model.num_classes. "
                f"Got {len(self.loss.class_weights)} vs {self.model.num_classes}."
            )

        if self.run_id is not None and not self.run_id:
            raise ValueError("run_id must be null or a non-empty string")


def _build_dataclass(dataclass_type, raw: Any):
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected mapping for {dataclass_type.__name__}")
    return dataclass_type(**dict(raw))


def _looks_like_flat_config(data: Mapping[str, Any]) -> bool:
    if any(section in data for section in NESTED_SECTIONS):
        return False
    return any(key in data for key in FLAT_KEYS)
