"""
Background ring and shell/core mask utilities for fluorescence analysis.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from skimage.morphology import disk, binary_dilation, binary_erosion


def compute_background_ring(
    mask: np.ndarray,
    dilation_r: int = 5,
    erosion_r: int = 3,
) -> np.ndarray:
    """Return a boolean ring around *mask* for local background estimation.

    The ring is defined as ``dilated_mask & ~original_mask``, with the
    inner boundary optionally eroded to keep a gap between the cell and
    the ring.

    Args:
        mask: Binary cell mask (H, W), any dtype truthy/falsy.
        dilation_r: Radius of the outer dilation (defines ring outer edge).
        erosion_r: Radius of an optional inner erosion applied to the
            original mask before subtraction.  Set to 0 to disable.

    Returns:
        Boolean (H, W) array – ``True`` inside the ring.
    """
    binary = mask.astype(bool)
    outer = binary_dilation(binary, disk(dilation_r))
    if erosion_r > 0:
        inner = binary_dilation(binary, disk(erosion_r))
    else:
        inner = binary
    ring = outer & ~inner
    return ring


def background_stats(
    image: np.ndarray,
    ring_mask: np.ndarray,
) -> dict[str, float]:
    """Compute mean / median / std of *image* inside *ring_mask*.

    Args:
        image: Single-channel image (H, W).
        ring_mask: Boolean mask selecting background pixels.

    Returns:
        Dictionary with keys ``bg_mean``, ``bg_median``, ``bg_std``.
        All values are ``NaN`` when the ring contains no pixels.
    """
    pixels = image[ring_mask]
    if pixels.size == 0:
        logger.debug("Background ring is empty; returning NaN stats")
        return {"bg_mean": np.nan, "bg_median": np.nan, "bg_std": np.nan}
    return {
        "bg_mean": float(np.mean(pixels)),
        "bg_median": float(np.median(pixels)),
        "bg_std": float(np.std(pixels)),
    }


def compute_shell_core(
    mask: np.ndarray,
    erosion_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a binary mask into *shell* (cell-wall) and *core* (interior).

    Args:
        mask: Binary mask (H, W).
        erosion_radius: Structuring-element radius for the erosion that
            defines the core.

    Returns:
        ``(shell, core)`` – both boolean (H, W).
    """
    binary = mask.astype(bool)
    core = binary_erosion(binary, disk(erosion_radius))
    shell = binary & ~core
    return shell, core


def shell_core_stats(
    image: np.ndarray,
    shell: np.ndarray,
    core: np.ndarray,
) -> dict[str, float]:
    """Compute shell / core mean intensities and their ratio.

    Args:
        image: Single-channel image (H, W).
        shell: Boolean shell mask.
        core: Boolean core mask.

    Returns:
        Dictionary with ``shell_mean``, ``core_mean``,
        ``shell_core_ratio``.
    """
    shell_px = image[shell]
    core_px = image[core]

    shell_mean = float(np.mean(shell_px)) if shell_px.size > 0 else np.nan
    core_mean = float(np.mean(core_px)) if core_px.size > 0 else np.nan

    eps = 1e-10
    if np.isnan(shell_mean) or np.isnan(core_mean):
        ratio = np.nan
    else:
        ratio = shell_mean / (core_mean + eps)

    return {
        "shell_mean": shell_mean,
        "core_mean": core_mean,
        "shell_core_ratio": ratio,
    }
