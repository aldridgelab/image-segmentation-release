"""Padmap registration module for joining drug labels to cell metadata."""

from deciphaer_processing.padmap.registry import (
    load_padmap,
    derive_padmap_id,
    apply_padmap,
    PadmapMatchReport,
)

__all__ = [
    "load_padmap",
    "derive_padmap_id",
    "apply_padmap",
    "PadmapMatchReport",
]
