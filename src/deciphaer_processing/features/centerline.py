"""
Centerline (midline) feature extraction for single-cell analysis.

Extracts cell length, width, sinuosity, and curvature features
using skeletonization and contour analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from scipy import ndimage
from scipy.interpolate import splprep, splev
from skimage import measure, morphology


def extract_centerline_features(
    mask: np.ndarray,
    pixel_microns: float = 0.10317,
    label: int | None = None,
) -> dict[str, Any]:
    """
    Extract centerline-based features from a binary mask.

    Based on MOMIA2's fast_midline() and direct_intersect_distance() functions.

    Args:
        mask: Binary mask (H, W) or labeled mask with specific label
        pixel_microns: Pixel size in microns for unit conversion
        label: If mask is labeled, extract features for this specific label

    Returns:
        Dictionary of feature name -> value
    """
    # Ensure binary mask
    if label is not None:
        binary_mask = (mask == label).astype(np.uint8)
    else:
        binary_mask = (mask > 0).astype(np.uint8)

    # Label and take largest connected component
    labeled_mask, n_labels = ndimage.label(binary_mask)
    if n_labels == 0:
        logger.warning("No objects found in mask")
        return _empty_centerline_features()

    if n_labels > 1:
        sizes = ndimage.sum(binary_mask, labeled_mask, range(1, n_labels + 1))
        largest_label = np.argmax(sizes) + 1
        binary_mask = (labeled_mask == largest_label).astype(np.uint8)

    # Get contour
    contours = measure.find_contours(binary_mask.astype(np.float32, copy=False), 0.5)
    if len(contours) == 0:
        logger.warning("No contours found")
        return _empty_centerline_features()

    contour = max(contours, key=len)  # Take longest contour

    # Get region properties for orientation and centroid
    props = measure.regionprops(binary_mask.astype(np.int32, copy=False))
    if len(props) == 0:
        return _empty_centerline_features()

    prop = props[0]
    centroid = prop.centroid
    orientation = prop.orientation

    # Compute midline
    try:
        midline = _fast_midline(
            binary_mask,
            contour,
            centroid=centroid,
            orientation=orientation,
            pixel_microns=pixel_microns,
        )
    except Exception as e:
        logger.warning(f"Midline computation failed: {e}")
        midline = None

    features = {}

    if midline is not None and len(midline) >= 3:
        # Length (arc length of midline)
        length_px = _measure_length(midline)
        features["length_px"] = length_px
        features["length_um"] = length_px * pixel_microns

        # Sinuosity (arc length / straight-line distance)
        straight_dist = np.sqrt(
            (midline[-1, 0] - midline[0, 0]) ** 2 +
            (midline[-1, 1] - midline[0, 1]) ** 2
        )
        if straight_dist > 0:
            features["sinuosity"] = length_px / straight_dist
        else:
            features["sinuosity"] = 1.0

        # Width measurements along midline
        widths = _direct_intersect_distance(midline, contour)
        if len(widths) > 0:
            # Filter out zeros at poles
            valid_widths = widths[widths > 0]
            if len(valid_widths) > 0:
                features["width_median_px"] = np.median(valid_widths)
                features["width_median_um"] = features["width_median_px"] * pixel_microns
                features["width_std_px"] = np.std(valid_widths)
                features["width_std_um"] = features["width_std_px"] * pixel_microns
                features["width_max_px"] = np.max(valid_widths)
                features["width_max_um"] = features["width_max_px"] * pixel_microns
                features["width_min_px"] = np.min(valid_widths)
                features["width_min_um"] = features["width_min_px"] * pixel_microns
                features["width_q1_px"] = np.percentile(valid_widths, 25)
                features["width_q1_um"] = features["width_q1_px"] * pixel_microns
                features["width_q3_px"] = np.percentile(valid_widths, 75)
                features["width_q3_um"] = features["width_q3_px"] * pixel_microns
            else:
                _add_empty_width_features(features)
        else:
            _add_empty_width_features(features)

        # Curvature along midline
        curvatures = _compute_curvature(midline)
        if len(curvatures) > 0:
            features["curvature_mean"] = np.mean(curvatures)
            features["curvature_std"] = np.std(curvatures)
            features["curvature_max"] = np.max(curvatures)
            features["curvature_min"] = np.min(curvatures)
        else:
            features["curvature_mean"] = np.nan
            features["curvature_std"] = np.nan
            features["curvature_max"] = np.nan
            features["curvature_min"] = np.nan
    else:
        # Fallback to major axis as length estimate
        features["length_px"] = prop.major_axis_length
        features["length_um"] = prop.major_axis_length * pixel_microns
        features["sinuosity"] = 1.0
        _add_empty_width_features(features)
        features["curvature_mean"] = np.nan
        features["curvature_std"] = np.nan
        features["curvature_max"] = np.nan
        features["curvature_min"] = np.nan

    return features


def _fast_midline(
    mask: np.ndarray,
    contour: np.ndarray,
    centroid: tuple[float, float],
    orientation: float,
    pixel_microns: float = 0.10317,
    upsampling_factor: int = 2,
) -> np.ndarray | None:
    """
    Compute cell midline using orientation-based initialization.

    Simplified version of MOMIA2's fast_midline().
    """
    x0, y0 = int(round(centroid[0])), int(round(centroid[1]))

    # Get line segment through centroid along orientation
    dx, dy = np.cos(orientation), np.sin(orientation)

    # Find intersection with contour
    intersect = _line_contour_intersection([x0, y0], [x0 + dx, y0 + dy], contour)
    if intersect is None or len(intersect) < 2:
        return None

    # Estimate interpolation size
    length_est = np.sqrt(
        (intersect[1, 0] - intersect[0, 0]) ** 2 +
        (intersect[1, 1] - intersect[0, 1]) ** 2
    )
    n_points = max(10, int(round(length_est * upsampling_factor)))

    # Linear interpolation for initial midline
    midline = np.array([
        np.linspace(intersect[0, 0], intersect[1, 0], n_points),
        np.linspace(intersect[0, 1], intersect[1, 1], n_points),
    ]).T

    # Smooth with spline if enough points
    if len(midline) >= 4:
        try:
            midline = _spline_smooth(midline, n=max(3, int(length_est)))
        except Exception:
            pass

    return midline


def _line_contour_intersection(
    p1: list[float],
    p2: list[float],
    contour: np.ndarray,
) -> np.ndarray | None:
    """Find intersection points of a line with a contour."""
    # Direction vector
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Extend line in both directions
    t_range = 100  # Extend by this factor
    line_start = np.array([p1[0] - t_range * dx, p1[1] - t_range * dy])
    line_end = np.array([p1[0] + t_range * dx, p1[1] + t_range * dy])

    intersections = []
    n = len(contour)

    for i in range(n):
        p3 = contour[i]
        p4 = contour[(i + 1) % n]

        # Line-segment intersection
        denom = (line_start[0] - line_end[0]) * (p3[1] - p4[1]) - \
                (line_start[1] - line_end[1]) * (p3[0] - p4[0])

        if abs(denom) < 1e-10:
            continue

        t = ((line_start[0] - p3[0]) * (p3[1] - p4[1]) -
             (line_start[1] - p3[1]) * (p3[0] - p4[0])) / denom
        u = -((line_start[0] - line_end[0]) * (line_start[1] - p3[1]) -
              (line_start[1] - line_end[1]) * (line_start[0] - p3[0])) / denom

        if 0 <= u <= 1:  # Intersection within contour segment
            ix = line_start[0] + t * (line_end[0] - line_start[0])
            iy = line_start[1] + t * (line_end[1] - line_start[1])
            intersections.append([ix, iy])

    if len(intersections) < 2:
        return None

    intersections = np.array(intersections)

    # Find the two furthest points (the actual cell endpoints)
    if len(intersections) == 2:
        return intersections

    # For multiple intersections, take the two furthest apart
    max_dist = 0
    best_pair = (0, 1)
    for i in range(len(intersections)):
        for j in range(i + 1, len(intersections)):
            dist = np.sqrt(
                (intersections[i, 0] - intersections[j, 0]) ** 2 +
                (intersections[i, 1] - intersections[j, 1]) ** 2
            )
            if dist > max_dist:
                max_dist = dist
                best_pair = (i, j)

    return intersections[list(best_pair)]


def _measure_length(line: np.ndarray) -> float:
    """Compute arc length of a line (sum of segment lengths)."""
    if len(line) < 2:
        return 0.0
    diffs = np.diff(line, axis=0)
    return np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))


def _direct_intersect_distance(midline: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Compute cell widths perpendicular to midline at each point.

    Simplified version of MOMIA2's direct_intersect_distance().
    """
    if len(midline) < 3:
        return np.array([])

    # Skip endpoints
    interior_midline = midline[1:-1]
    if len(interior_midline) == 0:
        return np.array([])

    # Compute perpendicular directions
    tangents = np.diff(midline, axis=0)
    tangents = tangents[:-1] + tangents[1:]  # Average at interior points
    tangent_norms = np.sqrt(np.sum(tangents ** 2, axis=1, keepdims=True))
    tangent_norms[tangent_norms == 0] = 1
    tangents = tangents / tangent_norms

    # Perpendicular vectors
    perps = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    widths = []
    for i, (point, perp) in enumerate(zip(interior_midline, perps)):
        # Cast ray in perpendicular direction
        dist1 = _ray_contour_distance(point, perp, contour)
        dist2 = _ray_contour_distance(point, -perp, contour)

        if dist1 is not None and dist2 is not None:
            widths.append(dist1 + dist2)
        else:
            widths.append(0)

    return np.array([0] + widths + [0])  # Add zeros at poles


def _ray_contour_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    contour: np.ndarray,
) -> float | None:
    """Find distance from origin to contour along a ray direction."""
    min_dist = float("inf")
    n = len(contour)

    for i in range(n):
        p1 = contour[i]
        p2 = contour[(i + 1) % n]

        # Ray-segment intersection
        denom = direction[0] * (p1[1] - p2[1]) - direction[1] * (p1[0] - p2[0])
        if abs(denom) < 1e-10:
            continue

        t = ((p1[0] - origin[0]) * (p1[1] - p2[1]) -
             (p1[1] - origin[1]) * (p1[0] - p2[0])) / denom
        u = ((p1[0] - origin[0]) * direction[1] -
             (p1[1] - origin[1]) * direction[0]) / denom

        if t > 0 and 0 <= u <= 1:  # Positive direction, within segment
            if t < min_dist:
                min_dist = t

    return min_dist if min_dist < float("inf") else None


def _compute_curvature(line: np.ndarray) -> np.ndarray:
    """Compute curvature along a line using finite differences."""
    if len(line) < 3:
        return np.array([])

    # First and second derivatives
    dx = np.gradient(line[:, 0])
    dy = np.gradient(line[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom[denom == 0] = 1e-10

    curvature = np.abs(dx * ddy - dy * ddx) / denom

    return curvature


def _spline_smooth(line: np.ndarray, n: int = None, s: float = 1.0) -> np.ndarray:
    """Smooth a line using B-spline interpolation."""
    if n is None:
        n = len(line)

    try:
        # Fit spline
        tck, u = splprep([line[:, 0], line[:, 1]], s=s, k=min(3, len(line) - 1))
        # Evaluate at n points
        u_new = np.linspace(0, 1, n)
        smoothed = np.array(splev(u_new, tck)).T
        return smoothed
    except Exception:
        return line


def _add_empty_width_features(features: dict) -> None:
    """Add NaN width features to a dictionary."""
    for suffix in ["median_px", "median_um", "std_px", "std_um",
                   "max_px", "max_um", "min_px", "min_um",
                   "q1_px", "q1_um", "q3_px", "q3_um"]:
        features[f"width_{suffix}"] = np.nan


def _empty_centerline_features() -> dict[str, Any]:
    """Return a dictionary of NaN features when no object is found."""
    features = {
        "length_px": np.nan,
        "length_um": np.nan,
        "sinuosity": np.nan,
        "curvature_mean": np.nan,
        "curvature_std": np.nan,
        "curvature_max": np.nan,
        "curvature_min": np.nan,
    }
    _add_empty_width_features(features)
    return features


def compute_midline(
    mask: np.ndarray,
    pixel_microns: float = 0.10317,
    label: int | None = None,
) -> np.ndarray | None:
    """Compute the midline for a cell mask, returning the raw coordinate array.

    This is a lightweight wrapper that extracts the midline without computing
    the full centerline feature set.  Used by the processing pipeline to pass
    the midline to intensity / fluorescence feature extraction.

    Args:
        mask: Binary or labelled mask (H, W).
        pixel_microns: Pixel size in microns.
        label: If mask is labelled, extract midline for this label.

    Returns:
        (N, 2) array of midline coordinates (row, col), or ``None`` on failure.
    """
    if label is not None:
        binary_mask = (mask == label).astype(np.uint8)
    else:
        binary_mask = (mask > 0).astype(np.uint8)

    labeled_mask, n_labels = ndimage.label(binary_mask)
    if n_labels == 0:
        return None

    if n_labels > 1:
        sizes = ndimage.sum(binary_mask, labeled_mask, range(1, n_labels + 1))
        largest_label = np.argmax(sizes) + 1
        binary_mask = (labeled_mask == largest_label).astype(np.uint8)

    contours = measure.find_contours(binary_mask.astype(np.float32, copy=False), 0.5)
    if len(contours) == 0:
        return None

    contour = max(contours, key=len)
    props = measure.regionprops(binary_mask.astype(np.int32, copy=False))
    if len(props) == 0:
        return None

    prop = props[0]
    try:
        midline = _fast_midline(
            binary_mask, contour,
            centroid=prop.centroid,
            orientation=prop.orientation,
            pixel_microns=pixel_microns,
        )
        if midline is not None and len(midline) >= 3:
            return midline
    except Exception:
        pass
    return None
