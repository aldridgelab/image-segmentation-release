"""Resolve explicitly identified pixel-unit settings at an image calibration.

No image resampling is performed. Physical and dimensionless settings are
intentionally absent from the conversion tables below.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping


def spatial_scale(pixel_microns: float, reference_pixel_microns: float | None) -> float:
    for name, value in (
        ("inputs.pixel_microns", pixel_microns),
        ("spatial_calibration.reference_pixel_microns", reference_pixel_microns),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and > 0")
    factor = 1.0 if reference_pixel_microns is None else reference_pixel_microns / pixel_microns
    if not math.isfinite(factor * factor) or factor * factor <= 0:
        raise ValueError("Spatial calibration ratio must have a finite, positive square")
    return factor


def scaled_integer(value: int, factor: float, *, minimum: int = 0, odd: bool = False) -> int:
    """Round half up; odd windows round to the nearest odd integer (ties up)."""
    target = value * factor
    if odd:
        return max(minimum, 2 * math.floor(target / 2) + 1)
    return max(minimum, math.floor(target + 0.5))


def scale_momia_settings(settings: Mapping[str, Any], factor: float) -> dict[str, Any]:
    """Scale a merged, validated settings mapping without mutating its source."""
    result = copy.deepcopy(dict(settings))
    if factor == 1:
        return result
    for section, key in (
        ("filtering", "edge_distance"),
        ("centrality", "center_window_px"),
    ):
        result[section][key] = scaled_integer(int(result[section][key]), factor)
    result["image"]["max_drift"] = float(result["image"]["max_drift"]) * factor
    result["segmentation"]["window_size"] = scaled_integer(
        int(result["segmentation"]["window_size"]), factor, minimum=3, odd=True,
    )
    for key in ("min_particle_size", "min_hole_size"):
        result["segmentation"][key] = scaled_integer(int(result["segmentation"][key]), factor * factor)
    for key in ("area_min", "area_max"):
        if result["filtering"][key] is not None:
            result["filtering"][key] = float(result["filtering"][key]) * factor * factor
    return result
