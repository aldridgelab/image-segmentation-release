"""
Intensity feature extraction for single-cell analysis.

Extracts per-channel intensity statistics and profile features
along the cell midline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from scipy import ndimage, stats


# Channel names in standard order
CHANNEL_NAMES = ["Phase", "AF405", "Bod493"]


def extract_intensity_features(
    mask: np.ndarray,
    image: np.ndarray,
    channel_names: list[str] | None = None,
    midline: np.ndarray | None = None,
    label: int | None = None,
) -> dict[str, Any]:
    """
    Extract intensity features from image channels within a mask.

    Args:
        mask: Binary mask (H, W) or labeled mask with specific label
        image: Multi-channel image (C, H, W) or single channel (H, W)
        channel_names: Names for each channel (default: Phase, AF405, Bod493)
        midline: Optional midline array for profile features
        label: If mask is labeled, extract features for this specific label

    Returns:
        Dictionary of feature name -> value
    """
    # Ensure binary mask
    if label is not None:
        binary_mask = (mask == label).astype(bool)
    else:
        binary_mask = (mask > 0).astype(bool)

    # Handle image dimensions
    if image.ndim == 2:
        image = image[np.newaxis, ...]  # Add channel dimension

    n_channels = image.shape[0]
    if channel_names is None:
        if n_channels == 3:
            channel_names = CHANNEL_NAMES
        else:
            channel_names = [f"ch{i}" for i in range(n_channels)]

    features = {}

    # Extract features for each channel
    for c, channel_name in enumerate(channel_names):
        if c >= n_channels:
            break

        channel_img = image[c]
        channel_features = _extract_channel_intensity(
            binary_mask, channel_img, channel_name
        )
        features.update(channel_features)

        # Profile features along midline if available
        if midline is not None and len(midline) >= 3:
            profile_features = _extract_profile_features(
                channel_img, midline, channel_name
            )
            features.update(profile_features)

    return features


def _extract_channel_intensity(
    mask: np.ndarray,
    channel: np.ndarray,
    channel_name: str,
) -> dict[str, Any]:
    """
    Extract intensity statistics for a single channel within mask.

    Args:
        mask: Binary mask (H, W)
        channel: Single channel image (H, W)
        channel_name: Name prefix for features

    Returns:
        Dictionary of feature name -> value
    """
    prefix = f"intensity_{channel_name}"
    features = {}

    # Get masked values
    masked_values = channel[mask]

    if len(masked_values) == 0:
        logger.warning(f"No pixels in mask for channel {channel_name}")
        return _empty_channel_features(channel_name)

    # Basic statistics
    features[f"{prefix}_mean"] = np.mean(masked_values)
    features[f"{prefix}_median"] = np.median(masked_values)
    features[f"{prefix}_std"] = np.std(masked_values)
    features[f"{prefix}_var"] = np.var(masked_values)
    features[f"{prefix}_min"] = np.min(masked_values)
    features[f"{prefix}_max"] = np.max(masked_values)
    features[f"{prefix}_range"] = features[f"{prefix}_max"] - features[f"{prefix}_min"]

    # Percentiles
    features[f"{prefix}_q1"] = np.percentile(masked_values, 25)
    features[f"{prefix}_q3"] = np.percentile(masked_values, 75)
    features[f"{prefix}_iqr"] = features[f"{prefix}_q3"] - features[f"{prefix}_q1"]
    features[f"{prefix}_p5"] = np.percentile(masked_values, 5)
    features[f"{prefix}_p95"] = np.percentile(masked_values, 95)

    # Coefficient of variation
    mean_val = features[f"{prefix}_mean"]
    if mean_val != 0:
        features[f"{prefix}_cv"] = features[f"{prefix}_std"] / abs(mean_val)
    else:
        features[f"{prefix}_cv"] = np.nan

    # Distribution shape
    if len(masked_values) >= 4:
        features[f"{prefix}_skewness"] = stats.skew(masked_values)
        features[f"{prefix}_kurtosis"] = stats.kurtosis(masked_values)
    else:
        features[f"{prefix}_skewness"] = np.nan
        features[f"{prefix}_kurtosis"] = np.nan

    # Texture features (basic)
    features[f"{prefix}_entropy"] = _entropy(masked_values)

    return features


def _extract_profile_features(
    channel: np.ndarray,
    midline: np.ndarray,
    channel_name: str,
    width: int = 3,
) -> dict[str, Any]:
    """
    Extract intensity profile features along the midline.

    Based on MOMIA2's measure_along_strip().

    Args:
        channel: Single channel image (H, W)
        midline: Cell midline array (N, 2)
        channel_name: Name prefix for features
        width: Width of strip for profile measurement

    Returns:
        Dictionary of feature name -> value
    """
    prefix = f"profile_{channel_name}"
    features = {}

    # Measure intensity along strip
    profile = _measure_along_strip(midline, channel, width=width)

    if profile is None or len(profile) < 3:
        return _empty_profile_features(channel_name)

    # Profile statistics
    features[f"{prefix}_mean"] = np.mean(profile)
    features[f"{prefix}_std"] = np.std(profile)
    features[f"{prefix}_min"] = np.min(profile)
    features[f"{prefix}_max"] = np.max(profile)
    features[f"{prefix}_range"] = features[f"{prefix}_max"] - features[f"{prefix}_min"]

    # Gradient features (intensity changes along length)
    gradient = np.gradient(profile)
    features[f"{prefix}_gradient_mean"] = np.mean(np.abs(gradient))
    features[f"{prefix}_gradient_max"] = np.max(np.abs(gradient))

    # Polar asymmetry (compare first half to second half)
    mid_idx = len(profile) // 2
    first_half = profile[:mid_idx]
    second_half = profile[mid_idx:]
    if len(first_half) > 0 and len(second_half) > 0:
        features[f"{prefix}_polar_ratio"] = np.mean(first_half) / (np.mean(second_half) + 1e-10)
        features[f"{prefix}_polar_diff"] = np.mean(first_half) - np.mean(second_half)
    else:
        features[f"{prefix}_polar_ratio"] = np.nan
        features[f"{prefix}_polar_diff"] = np.nan

    # Peak analysis
    peak_idx = np.argmax(profile)
    features[f"{prefix}_peak_position"] = peak_idx / len(profile)  # Normalized position
    features[f"{prefix}_peak_value"] = profile[peak_idx]

    return features


def _measure_along_strip(
    line: np.ndarray,
    img: np.ndarray,
    width: int = 3,
    mode: str = "mean",
) -> np.ndarray | None:
    """
    Measure intensity along a strip perpendicular to a line.

    Simplified version of MOMIA2's measure_along_strip().

    Args:
        line: Line coordinates (N, 2)
        img: Image array (H, W)
        width: Width of strip perpendicular to line
        mode: Aggregation mode ('mean', 'max', 'min', 'median')

    Returns:
        Array of intensity values along the line
    """
    if len(line) < 2:
        return None

    h, w = img.shape

    # Compute perpendicular unit vectors
    tangents = np.diff(line, axis=0)
    tangents = np.vstack([tangents[0], tangents])  # Pad to same length
    tangent_norms = np.sqrt(np.sum(tangents ** 2, axis=1, keepdims=True))
    tangent_norms[tangent_norms == 0] = 1
    tangents = tangents / tangent_norms

    # Perpendicular vectors
    perps = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # Sample intensity along strip at each point
    profile = []
    half_width = width // 2

    for i, (point, perp) in enumerate(zip(line, perps)):
        values = []
        for offset in range(-half_width, half_width + 1):
            sample_point = point + offset * perp
            # Bilinear interpolation
            val = _bilinear_sample(img, sample_point[0], sample_point[1])
            if val is not None:
                values.append(val)

        if len(values) > 0:
            if mode == "max":
                profile.append(np.max(values))
            elif mode == "min":
                profile.append(np.min(values))
            elif mode == "median":
                profile.append(np.median(values))
            else:
                profile.append(np.mean(values))
        else:
            profile.append(np.nan)

    return np.array(profile)


def _bilinear_sample(img: np.ndarray, y: float, x: float) -> float | None:
    """Bilinear interpolation at a floating point coordinate."""
    h, w = img.shape

    # Bounds check
    if y < 0 or y >= h - 1 or x < 0 or x >= w - 1:
        return None

    y0, x0 = int(y), int(x)
    y1, x1 = y0 + 1, x0 + 1

    # Interpolation weights
    wy = y - y0
    wx = x - x0

    # Bilinear interpolation
    val = (1 - wy) * (1 - wx) * img[y0, x0] + \
          (1 - wy) * wx * img[y0, x1] + \
          wy * (1 - wx) * img[y1, x0] + \
          wy * wx * img[y1, x1]

    return val


def _entropy(values: np.ndarray, n_bins: int = 64) -> float:
    """Compute Shannon entropy of intensity values."""
    if len(values) == 0:
        return np.nan

    # Histogram
    hist, _ = np.histogram(values, bins=n_bins, density=True)
    hist = hist[hist > 0]  # Remove zeros

    if len(hist) == 0:
        return 0.0

    # Shannon entropy
    return -np.sum(hist * np.log2(hist + 1e-10))


def _empty_channel_features(channel_name: str) -> dict[str, Any]:
    """Return NaN features for a channel."""
    prefix = f"intensity_{channel_name}"
    suffixes = [
        "mean", "median", "std", "var", "min", "max", "range",
        "q1", "q3", "iqr", "p5", "p95", "cv", "skewness", "kurtosis", "entropy"
    ]
    return {f"{prefix}_{s}": np.nan for s in suffixes}


def _empty_profile_features(channel_name: str) -> dict[str, Any]:
    """Return NaN profile features for a channel."""
    prefix = f"profile_{channel_name}"
    suffixes = [
        "mean", "std", "min", "max", "range",
        "gradient_mean", "gradient_max",
        "polar_ratio", "polar_diff",
        "peak_position", "peak_value"
    ]
    return {f"{prefix}_{s}": np.nan for s in suffixes}
