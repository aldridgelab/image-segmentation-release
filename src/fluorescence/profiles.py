"""
Midline intensity profile computation and scalar summary extraction.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.signal import find_peaks

from momia2.utils.midline import measure_along_strip
from momia2.plot.plot_data import reorient_data


def compute_midline_profile(
    midline: np.ndarray,
    image: np.ndarray,
    width: int = 3,
    mode: str = "mean",
    n_points: int = 100,
) -> np.ndarray | None:
    """Sample intensity along a midline and resample to a fixed length.

    Uses ``momia2.utils.midline.measure_along_strip`` for the raw profile,
    then linearly interpolates to *n_points*.

    Args:
        midline: (N, 2) array of midline coordinates (row, col).
        image: Single-channel image (H, W).
        width: Perpendicular strip width in pixels.
        mode: Aggregation across the strip (``"mean"`` | ``"max"``
            | ``"min"`` | ``"median"``).
        n_points: Length of the resampled output profile.

    Returns:
        1-D array of length *n_points*, or ``None`` on failure.
    """
    if midline is None or len(midline) < 3:
        return None
    try:
        raw = measure_along_strip(midline, image, width=width, mode=mode)
        if raw is None or len(raw) < 3:
            return None
        resampled = np.interp(
            np.linspace(0, 1, n_points),
            np.linspace(0, 1, len(raw)),
            raw,
        )
        return resampled
    except Exception as exc:
        logger.debug(f"Profile computation failed: {exc}")
        return None


def reorient_profile(profile: np.ndarray) -> tuple[np.ndarray, bool]:
    """Reorient a 1-D profile so the brighter pole is always first.

    Wraps ``momia2.plot.plot_data.reorient_data``.

    Args:
        profile: 1-D intensity array.

    Returns:
        ``(reoriented_profile, was_flipped)``
    """
    return reorient_data(profile)


def profile_scalar_summaries(profile: np.ndarray) -> dict[str, float]:
    """Derive scalar features from a 1-D midline intensity profile.

    Features extracted:
    - ``midline_mean``, ``midline_max``
    - ``midline_gradient_mean``, ``midline_gradient_max``
      (mean / max of the absolute first-derivative)
    - ``midline_peak_position`` (normalised 0-1 position of the global max)
    - ``midline_peak_value``

    Args:
        profile: 1-D intensity array (already resampled).

    Returns:
        Dictionary of scalar feature values.  All ``NaN`` when
        *profile* is ``None`` or too short.
    """
    nan_result: dict[str, float] = {
        "midline_mean": np.nan,
        "midline_max": np.nan,
        "midline_gradient_mean": np.nan,
        "midline_gradient_max": np.nan,
        "midline_peak_position": np.nan,
        "midline_peak_value": np.nan,
    }
    if profile is None or len(profile) < 3:
        return nan_result

    grad = np.abs(np.diff(profile))

    peak_idx = int(np.argmax(profile))
    peak_pos = peak_idx / max(len(profile) - 1, 1)

    return {
        "midline_mean": float(np.mean(profile)),
        "midline_max": float(np.max(profile)),
        "midline_gradient_mean": float(np.mean(grad)),
        "midline_gradient_max": float(np.max(grad)),
        "midline_peak_position": float(peak_pos),
        "midline_peak_value": float(profile[peak_idx]),
    }
