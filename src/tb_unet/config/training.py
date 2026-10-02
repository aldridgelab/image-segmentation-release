"""Training runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SchedulerName = Literal["cosine", "plateau", "step", "none"]


@dataclass
class TrainingConfig:
    """Optimizer/scheduler/training-loop settings."""

    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: SchedulerName = "cosine"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 20
    mixed_precision: bool = True
    gradient_clip: float = 1.0

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("training.batch_size must be >= 1")
        if self.num_workers < 0:
            raise ValueError("training.num_workers must be >= 0")
        if self.epochs < 1:
            raise ValueError("training.epochs must be >= 1")
        if self.lr <= 0:
            raise ValueError("training.lr must be > 0")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay must be >= 0")
        if self.scheduler not in {"cosine", "plateau", "step", "none"}:
            raise ValueError("training.scheduler must be one of: cosine, plateau, step, none")
        if self.scheduler_patience < 1:
            raise ValueError("training.scheduler_patience must be >= 1")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("training.scheduler_factor must be in (0, 1)")
        if self.early_stopping_patience < 1:
            raise ValueError("training.early_stopping_patience must be >= 1")
        if self.gradient_clip < 0:
            raise ValueError("training.gradient_clip must be >= 0")
