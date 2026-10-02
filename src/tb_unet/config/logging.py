"""Experiment logging configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoggingConfig:
    """External logging/backends (W&B) settings."""

    wandb_enabled: bool = False
    wandb_project: str = "tb-unet"
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.wandb_project:
            raise ValueError("logging.wandb_project must not be empty")
        if any(not tag for tag in self.wandb_tags):
            raise ValueError("logging.wandb_tags must not contain empty values")
