"""
HADA (AF405) polar / septal labelling feature extraction.

Computes features describing the spatial distribution of cell-wall label
along the midline: pole intensities, centre enrichment, asymmetry, and
a simple discrete state classification.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.signal import find_peaks


def hada_polar_features(
    profile: np.ndarray,
    pole_window: float = 0.10,
    center_window: tuple[float, float] = (0.45, 0.55),
) -> dict[str, float]:
    """Extract HADA polar scalar features from a resampled midline profile.

    The profile should already be background-subtracted and reoriented
    (brighter pole first) before calling this function.

    Args:
        profile: 1-D intensity array of length N (e.g. 100 points).
        pole_window: Fraction of profile length assigned to each pole.
        center_window: ``(start, end)`` fractions defining the centre
            region.

    Returns:
        Dictionary with the following keys (all ``NaN`` when profile
        is invalid):

        - ``poleA``, ``poleB``, ``center`` – mean intensities.
        - ``polar_asym`` – ``|poleA - poleB| / (poleA + poleB + eps)``
        - ``center_enrichment`` – ``center / (mean(poles) + eps)``
        - ``peak_count`` – number of peaks detected on a smoothed profile.
    """
    nan_result: dict[str, float] = {
        "poleA": np.nan,
        "poleB": np.nan,
        "center": np.nan,
        "polar_asym": np.nan,
        "center_enrichment": np.nan,
        "peak_count": np.nan,
    }
    if profile is None or len(profile) < 10:
        return nan_result

    n = len(profile)
    eps = 1e-10

    # Window indices
    pole_end = max(1, int(round(n * pole_window)))
    center_start = int(round(n * center_window[0]))
    center_end = int(round(n * center_window[1]))

    poleA = float(np.mean(profile[:pole_end]))
    poleB = float(np.mean(profile[-pole_end:]))
    center = float(np.mean(profile[center_start:center_end]))

    polar_asym = abs(poleA - poleB) / (poleA + poleB + eps)
    mean_poles = 0.5 * (poleA + poleB)
    center_enrichment = center / (mean_poles + eps)

    # Simple peak count on smoothed profile
    try:
        smoothed = np.convolve(profile, np.ones(max(1, n // 20)) / max(1, n // 20), mode="same")
        # Peaks must be at least 10% of max above baseline
        height_thresh = smoothed.min() + 0.10 * (smoothed.max() - smoothed.min())
        peaks, _ = find_peaks(smoothed, height=height_thresh, distance=max(1, n // 10))
        peak_count = float(len(peaks))
    except Exception:
        peak_count = np.nan

    return {
        "poleA": poleA,
        "poleB": poleB,
        "center": center,
        "polar_asym": polar_asym,
        "center_enrichment": center_enrichment,
        "peak_count": peak_count,
    }


def hada_state_classify(
    polar_asym: float,
    center_enrichment: float,
    asym_threshold: float = 0.3,
    center_threshold: float = 1.5,
) -> tuple[int, str]:
    """Classify HADA labelling pattern into a discrete state.

    States:
        0 – ``"none"``     (low signal everywhere)
        1 – ``"one_pole"`` (high asymmetry, low centre)
        2 – ``"two_poles"``(low asymmetry, low centre)
        3 – ``"septal"``   (high centre enrichment)

    Args:
        polar_asym: Polar asymmetry value.
        center_enrichment: Centre enrichment value.
        asym_threshold: Asymmetry above this -> one-pole bias.
        center_threshold: Centre enrichment above this -> septal.

    Returns:
        ``(state_code, state_label)``
    """
    if np.isnan(polar_asym) or np.isnan(center_enrichment):
        return -1, "unknown"

    if center_enrichment >= center_threshold:
        return 3, "septal"
    if polar_asym >= asym_threshold:
        return 1, "one_pole"
    # Both poles labelled and centre not enriched
    return 2, "two_poles"
