"""
Bod493 puncta (lipid body) detection – wraps momia2.utils.fish.find_puncta.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from skimage import feature, filters

import scipy.ndimage as _ndi

# momia2.utils.fish.find_puncta uses ``ndi`` at module scope but the
# wildcard import that normally supplies it may be missing in some builds.
import momia2.utils.fish as _fish_mod
if not hasattr(_fish_mod, "ndi"):
    _fish_mod.ndi = _ndi  # type: ignore[attr-defined]


def detect_puncta(
    image: np.ndarray,
    mask: np.ndarray,
    core_mask: np.ndarray | None = None,
    min_distance: int = 1,
    threshold_rel: float = 0.1,
    gaussian_sigma: float = 0.5,
    log_sigma: float = 1.0,
    threshold_abs: float | None = None,
    exclude_border: bool = True,
    bg_value: float | None = None,
) -> np.ndarray:
    """Detect fluorescent puncta inside a cell.

    Wraps ``momia2.utils.fish.find_puncta`` with optional background
    subtraction and core-masking.

    Args:
        image: Single-channel fluorescence image (H, W).
        mask: Binary cell mask (H, W).
        core_mask: Optional eroded mask restricting detection to the cell
            interior (avoids boundary artefacts).
        min_distance: Minimum distance between puncta centres (pixels).
        threshold_rel: Relative intensity threshold for the LoG peak
            finder.
        gaussian_sigma: Gaussian smoothing sigma (pixels) before LoG.
        log_sigma: Sigma for Laplacian-of-Gaussian peak map.
        threshold_abs: Optional absolute threshold for peak detection on
            the ``-LoG`` response map.
        exclude_border: Passed to ``skimage.feature.peak_local_max``.
        bg_value: If provided, subtract this value from *image* before
            detection.

    Returns:
        (K, 4) array with columns ``[y_subpx, x_subpx, raw_intensity,
        LoG_value]``.  Empty ``(0, 4)`` array when no puncta are found.
    """
    work_img = image.astype(np.float64, copy=True)
    if bg_value is not None:
        work_img -= bg_value
        work_img[work_img < 0] = 0

    detection_mask = mask.astype(bool)
    if core_mask is not None:
        detection_mask = detection_mask & core_mask.astype(bool)

    # Ensure mask is not empty
    if not detection_mask.any():
        return np.empty((0, 4), dtype=np.float64)

    try:
        # Match MOMIA behavior but expose sigmas/threshold controls.
        img_smoothed = filters.gaussian(
            work_img,
            sigma=float(gaussian_sigma),
            preserve_range=True,
        )
        vmin, vmax = float(np.min(img_smoothed)), float(np.max(img_smoothed))
        if vmax > vmin:
            img_norm = (img_smoothed - vmin) / (vmax - vmin)
        else:
            img_norm = np.zeros_like(img_smoothed, dtype=np.float64)

        log_map = _ndi.gaussian_laplace(img_norm, sigma=float(log_sigma))
        response = (-log_map) * detection_mask.astype(np.float64)

        peak_kwargs: dict[str, float | int | bool] = {
            "min_distance": max(1, int(min_distance)),
            "threshold_rel": float(threshold_rel),
            "exclude_border": bool(exclude_border),
        }
        if threshold_abs is not None:
            peak_kwargs["threshold_abs"] = float(threshold_abs)

        extrema = feature.peak_local_max(response, **peak_kwargs)
        if extrema is None or len(extrema) == 0:
            return np.empty((0, 4), dtype=np.float64)

        output: list[list[float]] = []
        h, w = img_norm.shape
        for ex in extrema:
            x, y = int(ex[0]), int(ex[1])
            if x <= 0 or y <= 0 or x >= (h - 1) or y >= (w - 1):
                # Subpixel refinement needs a 3x3 neighborhood.
                continue

            newx, newy, stable_maxima = _fish_mod.subpixel_approximation_quadratic(
                x,
                y,
                img_norm,
            )
            if stable_maxima:
                output.append([float(newx), float(newy), float(work_img[x, y]), float(log_map[x, y])])

        if len(output) == 0:
            return np.empty((0, 4), dtype=np.float64)
        return np.asarray(output, dtype=np.float64)
    except Exception as exc:
        logger.debug(f"Puncta detection failed: {exc}")
        return np.empty((0, 4), dtype=np.float64)


def puncta_scalar_summary(
    puncta: np.ndarray,
    cell_total_intensity: float,
) -> dict[str, float]:
    """Compute scalar summary statistics from a puncta table.

    Args:
        puncta: (K, 4) array from :func:`detect_puncta`.  Columns are
            ``[y, x, intensity, LoG_value]``.
        cell_total_intensity: Total (or sum) intensity of the cell in
            the same channel, used for ``puncta_fraction``.

    Returns:
        Dictionary of scalar features.  All ``NaN`` when *puncta* is
        empty.
    """
    nan_result: dict[str, float] = {
        "puncta_count": 0.0,
        "puncta_total_intensity": np.nan,
        "puncta_max_intensity": np.nan,
        "puncta_mean_sigma_px": np.nan,
        "puncta_max_sigma_px": np.nan,
        "puncta_fraction": np.nan,
    }
    if puncta.size == 0:
        return nan_result

    intensities = puncta[:, 2]
    # LoG values give a proxy for sigma (spot size)
    log_vals = np.abs(puncta[:, 3])

    eps = 1e-10
    total_int = float(np.sum(intensities))

    return {
        "puncta_count": float(len(puncta)),
        "puncta_total_intensity": total_int,
        "puncta_max_intensity": float(np.max(intensities)),
        "puncta_mean_sigma_px": float(np.mean(log_vals)),
        "puncta_max_sigma_px": float(np.max(log_vals)),
        "puncta_fraction": total_int / (cell_total_intensity + eps),
    }


def puncta_axial_radial(
    puncta: np.ndarray,
    midline: np.ndarray | None,
) -> np.ndarray:
    """Compute normalised axial (along midline) position for each punctum.

    Args:
        puncta: (K, 4) array from :func:`detect_puncta`.
        midline: (N, 2) array of midline coordinates (row, col).

    Returns:
        (K, 2) array with columns ``[axial_s, radial_r]``.
        ``axial_s`` is in ``[0, 1]`` (projection onto midline).
        ``radial_r`` is the perpendicular distance in pixels (not normalised).
        Returns empty ``(0, 2)`` when inputs are invalid.
    """
    if puncta.size == 0 or midline is None or len(midline) < 2:
        return np.empty((0, 2), dtype=np.float64)

    # Cumulative arc-length along midline
    diffs = np.diff(midline, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]
    if total_length == 0:
        return np.empty((0, 2), dtype=np.float64)

    results = np.empty((len(puncta), 2), dtype=np.float64)

    for i, row in enumerate(puncta):
        pt = np.array([row[0], row[1]])

        # Find closest midline point
        dists = np.sqrt(np.sum((midline - pt) ** 2, axis=1))
        closest_idx = int(np.argmin(dists))

        axial_s = cum_length[closest_idx] / total_length
        radial_r = dists[closest_idx]

        results[i, 0] = axial_s
        results[i, 1] = radial_r

    return results
