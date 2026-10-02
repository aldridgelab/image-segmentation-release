"""Output and checkpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutputConfig:
    """Output paths and logging frequency settings."""

    output_dir: str = "tb_unet/checkpoints"
    experiment_name: str = "tb_unet_v1"
    save_every: int = 10
    log_every: int = 10

    def validate(self) -> None:
        if not self.output_dir:
            raise ValueError("output.output_dir must not be empty")
        if not self.experiment_name:
            raise ValueError("output.experiment_name must not be empty")
        if self.save_every < 1:
            raise ValueError("output.save_every must be >= 1")
        if self.log_every < 1:
            raise ValueError("output.log_every must be >= 1")
