"""Data loading configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Dataset paths and channel selection settings."""

    data_dir: str = "session_output"
    image_dir: str = "data/images"
    channels: list[int] = field(default_factory=lambda: [0])
    pad_multiple: int = 32

    def validate(self) -> None:
        if not self.data_dir:
            raise ValueError("data.data_dir must not be empty")
        if not self.image_dir:
            raise ValueError("data.image_dir must not be empty")
        if not self.channels:
            raise ValueError("data.channels must contain at least one channel index")
        if any((not isinstance(c, int)) or c < 0 for c in self.channels):
            raise ValueError("data.channels must be a list of non-negative integers")
        if self.pad_multiple < 1:
            raise ValueError("data.pad_multiple must be >= 1")
