"""
Region properties feature extraction for single-cell analysis.

Extracts MOMIA2-style morphological features from binary masks using
scikit-image regionprops.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from scipy import ndimage
from skimage import measure


def extract_regionprops(
    mask: np.ndarray,
    intensity_image: np.ndarray | None = None,
    pixel_microns: float = 0.10317,
    label: int | None = None,
) -> dict[str, Any]:
    """
    Extract region properties from a binary or labeled mask.

    Based on MOMIA2's get_regionprop() function with additional derived features.

    Args:
        mask: Binary mask (H, W) or labeled mask with specific label
        intensity_image: Optional intensity image for intensity-weighted features
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

    # Label the mask
    labeled_mask, n_labels = ndimage.label(binary_mask)

    if n_labels == 0:
        logger.warning("No objects found in mask")
        return _empty_features()

    # Take largest connected component if multiple
    if n_labels > 1:
        sizes = ndimage.sum(binary_mask, labeled_mask, range(1, n_labels + 1))
        largest_label = np.argmax(sizes) + 1
        labeled_mask = (labeled_mask == largest_label).astype(np.uint8)

    # Define properties to extract
    regionprop_properties = [
        "label",
        "bbox",
        "centroid",
        "area",
        "convex_area",
        "filled_area",
        "eccentricity",
        "solidity",
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "equivalent_diameter",
        "extent",
        "orientation",
        "moments_hu",
    ]

    # Add intensity properties if intensity image provided
    if intensity_image is not None:
        regionprop_properties.extend([
            "mean_intensity",
            "max_intensity",
            "min_intensity",
        ])
        # Use first channel if multi-channel
        if intensity_image.ndim == 3:
            intensity_for_props = intensity_image[0]
        else:
            intensity_for_props = intensity_image
    else:
        intensity_for_props = (labeled_mask > 0).astype(np.float32)

    labeled_mask = labeled_mask.astype(np.int32, copy=False)
    intensity_for_props = intensity_for_props.astype(np.float32, copy=False)

    # Extract regionprops
    props = measure.regionprops_table(
        labeled_mask,
        intensity_image=intensity_for_props,
        properties=regionprop_properties,
    )

    # Build feature dictionary
    features = {}

    # Basic morphology
    area_px = props["area"][0]
    perimeter_px = props["perimeter"][0]
    major_axis_px = props["major_axis_length"][0]
    minor_axis_px = props["minor_axis_length"][0]

    features["area_px"] = area_px
    features["area_um2"] = area_px * (pixel_microns ** 2)
    features["perimeter_px"] = perimeter_px
    features["perimeter_um"] = perimeter_px * pixel_microns
    features["major_axis_px"] = major_axis_px
    features["major_axis_um"] = major_axis_px * pixel_microns
    features["minor_axis_px"] = minor_axis_px
    features["minor_axis_um"] = minor_axis_px * pixel_microns

    # Shape descriptors
    features["eccentricity"] = props["eccentricity"][0]
    features["solidity"] = props["solidity"][0]
    features["extent"] = props["extent"][0]
    features["orientation"] = props["orientation"][0]

    # Derived features
    features["convex_area_px"] = props["convex_area"][0]
    features["filled_area_px"] = props["filled_area"][0]
    features["equivalent_diameter_px"] = props["equivalent_diameter"][0]
    features["equivalent_diameter_um"] = props["equivalent_diameter"][0] * pixel_microns

    # Compactness (circularity measure)
    # Compactness = P^2 / (4 * pi * A)
    # Equals 1 for a circle, >1 for elongated shapes
    if area_px > 0:
        features["compactness"] = (perimeter_px ** 2) / (4 * np.pi * area_px)
        features["circularity"] = (4 * np.pi * area_px) / (perimeter_px ** 2)
    else:
        features["compactness"] = np.nan
        features["circularity"] = np.nan

    # Rough length estimate
    # From MOMIA2: rough_length = (P - sqrt(max(P^2 - 16*A, 0))) / 4
    discriminant = max(perimeter_px ** 2 - 16 * area_px, 0)
    features["rough_length_px"] = (perimeter_px - np.sqrt(discriminant)) / 4
    features["rough_length_um"] = features["rough_length_px"] * pixel_microns

    # Aspect ratio
    if major_axis_px > 0:
        features["aspect_ratio"] = round(minor_axis_px / major_axis_px, 3)
    else:
        features["aspect_ratio"] = np.nan

    # Hu moments (7 invariant moments)
    hu_moments = props["moments_hu-0"][0] if "moments_hu-0" in props else None
    if hu_moments is not None:
        for i in range(7):
            features[f"hu_moment_{i}"] = props[f"moments_hu-{i}"][0]
    else:
        # Try alternative key format
        try:
            for i in range(7):
                features[f"hu_moment_{i}"] = props["moments_hu"][0][i]
        except (KeyError, IndexError):
            for i in range(7):
                features[f"hu_moment_{i}"] = np.nan

    # Centroid and bbox
    features["centroid_row"] = props["centroid-0"][0]
    features["centroid_col"] = props["centroid-1"][0]
    features["bbox_min_row"] = props["bbox-0"][0]
    features["bbox_min_col"] = props["bbox-1"][0]
    features["bbox_max_row"] = props["bbox-2"][0]
    features["bbox_max_col"] = props["bbox-3"][0]

    # Intensity features if available
    if intensity_image is not None:
        features["mean_intensity"] = props["mean_intensity"][0]
        features["max_intensity"] = props["max_intensity"][0]
        features["min_intensity"] = props["min_intensity"][0]

    return features


def _empty_features() -> dict[str, Any]:
    """Return a dictionary of NaN features when no object is found."""
    feature_names = [
        "area_px", "area_um2", "perimeter_px", "perimeter_um",
        "major_axis_px", "major_axis_um", "minor_axis_px", "minor_axis_um",
        "eccentricity", "solidity", "extent", "orientation",
        "convex_area_px", "filled_area_px", "equivalent_diameter_px", "equivalent_diameter_um",
        "compactness", "circularity", "rough_length_px", "rough_length_um",
        "aspect_ratio",
        "hu_moment_0", "hu_moment_1", "hu_moment_2", "hu_moment_3",
        "hu_moment_4", "hu_moment_5", "hu_moment_6",
        "centroid_row", "centroid_col",
        "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col",
    ]
    return {name: np.nan for name in feature_names}


def apply_morphology_filters(
    features: dict[str, Any],
    area_min: float | None = 100.0,
    area_max: float | None = 500.0,
    eccentricity_min: float | None = None,
    eccentricity_max: float | None = None,
    solidity_min: float | None = None,
    solidity_max: float | None = None,
    aspect_ratio_min: float | None = None,
    aspect_ratio_max: float | None = None,
) -> tuple[bool, list[str]]:
    """
    Apply morphology-based quality filters.

    Args:
        features: Feature dictionary from extract_regionprops
        area_min: Minimum area in pixels
        area_max: Maximum area in pixels
        eccentricity_min: Minimum eccentricity (0-1)
        eccentricity_max: Maximum eccentricity (0-1)
        solidity_min: Minimum solidity (0-1)
        solidity_max: Maximum solidity (0-1)
        aspect_ratio_min: Minimum aspect ratio
        aspect_ratio_max: Maximum aspect ratio

    Returns:
        Tuple of (passed: bool, failure_reasons: list[str])
    """
    failure_reasons = []
    area = features.get("area_px", np.nan)

    # Area filter
    if area_min is not None and area < area_min:
        failure_reasons.append(f"area_too_small ({area:.1f} < {area_min})")
    if area_max is not None and area > area_max:
        failure_reasons.append(f"area_too_large ({area:.1f} > {area_max})")

    # Eccentricity filter
    ecc = features.get("eccentricity", np.nan)
    if eccentricity_min is not None and ecc < eccentricity_min:
        failure_reasons.append(f"eccentricity_too_low ({ecc:.3f} < {eccentricity_min})")
    if eccentricity_max is not None and ecc > eccentricity_max:
        failure_reasons.append(f"eccentricity_too_high ({ecc:.3f} > {eccentricity_max})")

    # Solidity filter
    sol = features.get("solidity", np.nan)
    if solidity_min is not None and sol < solidity_min:
        failure_reasons.append(f"solidity_too_low ({sol:.3f} < {solidity_min})")
    if solidity_max is not None and sol > solidity_max:
        failure_reasons.append(f"solidity_too_high ({sol:.3f} > {solidity_max})")

    # Aspect ratio filter
    ar = features.get("aspect_ratio", np.nan)
    if aspect_ratio_min is not None and ar < aspect_ratio_min:
        failure_reasons.append(f"aspect_ratio_too_low ({ar:.3f} < {aspect_ratio_min})")
    if aspect_ratio_max is not None and ar > aspect_ratio_max:
        failure_reasons.append(f"aspect_ratio_too_high ({ar:.3f} > {aspect_ratio_max})")

    passed = len(failure_reasons) == 0
    return passed, failure_reasons
