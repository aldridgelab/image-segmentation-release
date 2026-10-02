"""
Batch MOMIA2 segmentation utilities.

This module provides a JSON-configurable batch runner that mirrors the
web-app MOMIA2 parameter surface (segmentation + filtering + centrality)
while producing session-style outputs:
    output/masks/<image_id>_mask.tif
    output/metadata/<image_id>_meta.json
"""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from loguru import logger
import numpy as np
import tifffile

DEFAULT_SETTINGS_FILENAME = "momia2_segmentation_settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "settings_version": 1,
    "io": {
        "input_dir": None,
        "output_dir": "session_output/momia2_batch",
        "pattern": "*.tif",
        "recursive": False,
    },
    "image": {
        "pixel_microns": 0.10317,
        "channel_names": ["Phase", "HADA", "Bodipy"],
    },
    "segmentation": {
        "method": "isodata",
        "min_particle_size": 50,
        "min_hole_size": 50,
        "dark_foreground": True,
        "window_size": 9,
        "k": 0.05,
    },
    "filtering": {
        "exclude_edge": True,
        "edge_distance": 8,
        "area_min": 100.0,
        "area_max": 500.0,
        "eccentricity_min": 0.0,
        "eccentricity_max": 1.0,
        "solidity_min": 0.0,
        "solidity_max": 1.0,
        "length_min": None,
        "length_max": None,
        "width_min": None,
        "width_max": None,
        "aspect_ratio_min": None,
        "aspect_ratio_max": None,
    },
    "centrality": {
        "enabled": False,
        "center_window_px": 20,
        "tie_breaker": "area",
        "fail_if_center_cell_filtered": True,
    },
    "runtime": {
        "workers": 1,
        "limit": None,
        "skip_existing": True,
        "force": False,
        "fail_fast": False,
        "dry_run": False,
        "log_level": "INFO",
    },
    "metadata": {
        "emit_ucid": True,
        "ucid_prefix": "m2_",
        "embed_effective_settings": True,
    },
}

_ALLOWED_METHODS = {"isodata", "sauvola", "local", "composite"}
_ALLOWED_TIE_BREAKERS = {"area", "eccentricity", "solidity"}

_ALLOWED_KEYS: dict[str, set[str]] = {
    "io": {"input_dir", "output_dir", "pattern", "recursive"},
    "image": {"pixel_microns", "channel_names"},
    "segmentation": {
        "method",
        "min_particle_size",
        "min_hole_size",
        "dark_foreground",
        "window_size",
        "k",
    },
    "filtering": {
        "exclude_edge",
        "edge_distance",
        "area_min",
        "area_max",
        "eccentricity_min",
        "eccentricity_max",
        "solidity_min",
        "solidity_max",
        "length_min",
        "length_max",
        "width_min",
        "width_max",
        "aspect_ratio_min",
        "aspect_ratio_max",
    },
    "centrality": {
        "enabled",
        "center_window_px",
        "tie_breaker",
        "fail_if_center_cell_filtered",
    },
    "runtime": {"workers", "limit", "skip_existing", "force", "fail_fast", "dry_run", "log_level"},
    "metadata": {"emit_ucid", "ucid_prefix", "embed_effective_settings"},
}

_TOP_LEVEL_KEYS = {"settings_version", *list(_ALLOWED_KEYS)}

_DEFAULT_CHANNEL_NAMES: tuple[str, ...] = ("Phase", "HADA", "Bodipy")
_MOMIA_METHOD_MAP: dict[str, int] = {
    "isodata": 1,
    "sauvola": 2,
    "local": 3,
    "composite": 5,
}


@dataclass
class ImageProcessResult:
    image_id: str
    source_path: str
    status: str
    n_cells_total: int | None = None
    n_cells_filtered: int | None = None
    mask_path: str | None = None
    metadata_path: str | None = None
    error: str | None = None


@dataclass
class BatchRunSummary:
    started_at: str
    finished_at: str
    input_dir: str
    output_dir: str
    pattern: str
    recursive: bool
    total_candidates: int
    limit_applied: int | None
    processed: int
    succeeded: int
    failed: int
    skipped: int
    dry_run: bool
    settings_path: str | None
    results: list[ImageProcessResult]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Deep-merge dictionaries.

    Dict values are merged recursively.
    Non-dict values replace the base value.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def load_app_compat_settings(
    user_settings_path: Path,
    app_config_path: Path,
) -> dict[str, Any]:
    """
    Load settings from web-app files (if they exist).

    Supported sources:
    - session_output/user_settings.json
    - session_output/app_config.json
    """
    layer: dict[str, Any] = {}

    if user_settings_path.exists():
        user_settings = load_json_file(user_settings_path)
        seg_map = {
            "method": "method",
            "min_particle_size": "min_particle_size",
            "min_hole_size": "min_hole_size",
            "dark_foreground": "dark_foreground",
            "window_size": "window_size",
            "k": "k",
        }
        filter_map = {
            "exclude_edge": "exclude_edge",
            "edge_distance": "edge_distance",
            "area_min": "area_min",
            "area_max": "area_max",
            "eccentricity_min": "eccentricity_min",
            "eccentricity_max": "eccentricity_max",
            "solidity_min": "solidity_min",
            "solidity_max": "solidity_max",
            "length_min": "length_min",
            "length_max": "length_max",
            "width_min": "width_min",
            "width_max": "width_max",
            "aspect_ratio_min": "aspect_ratio_min",
            "aspect_ratio_max": "aspect_ratio_max",
        }

        seg_layer = {dst: user_settings[src] for src, dst in seg_map.items() if src in user_settings}
        filter_layer = {dst: user_settings[src] for src, dst in filter_map.items() if src in user_settings}
        if seg_layer:
            layer["segmentation"] = seg_layer
        if filter_layer:
            layer["filtering"] = filter_layer

    if app_config_path.exists():
        app_config = load_json_file(app_config_path)
        io_layer: dict[str, Any] = {}
        image_layer: dict[str, Any] = {}

        if "input_dir" in app_config:
            io_layer["input_dir"] = app_config["input_dir"]
        if "output_dir" in app_config:
            io_layer["output_dir"] = app_config["output_dir"]
        if "input_pattern" in app_config:
            io_layer["pattern"] = app_config["input_pattern"]
        if "pixel_microns" in app_config:
            image_layer["pixel_microns"] = app_config["pixel_microns"]
        if "channel_names" in app_config:
            image_layer["channel_names"] = app_config["channel_names"]

        if io_layer:
            layer["io"] = deep_merge(layer.get("io", {}), io_layer)
        if image_layer:
            layer["image"] = deep_merge(layer.get("image", {}), image_layer)

    return layer


def merge_settings(
    *,
    defaults: Mapping[str, Any],
    app_settings: Mapping[str, Any] | None = None,
    json_settings: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Merge settings using precedence:
    defaults < app_settings < json_settings < cli_overrides.
    """
    merged = copy.deepcopy(dict(defaults))
    if app_settings:
        merged = deep_merge(merged, app_settings)
    if json_settings:
        merged = deep_merge(merged, json_settings)
    if cli_overrides:
        merged = deep_merge(merged, cli_overrides)
    return merged


def _assert_known_keys(settings: Mapping[str, Any]) -> None:
    unknown_top = set(settings) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(f"Unknown top-level settings keys: {sorted(unknown_top)}")

    for section, allowed in _ALLOWED_KEYS.items():
        section_val = settings.get(section)
        if section_val is None:
            continue
        if not isinstance(section_val, Mapping):
            raise ValueError(f"Settings section '{section}' must be an object")
        unknown = set(section_val) - allowed
        if unknown:
            raise ValueError(f"Unknown keys in '{section}': {sorted(unknown)}")


def _assert_numeric_range(name: str, low: float | int | None, high: float | int | None) -> None:
    if low is None or high is None:
        return
    if low > high:
        raise ValueError(f"Invalid range for '{name}': min={low} is greater than max={high}")


def validate_settings(settings: Mapping[str, Any]) -> None:
    """Validate settings shape and high-impact constraints."""
    _assert_known_keys(settings)

    io_cfg = settings.get("io", {})
    if io_cfg.get("input_dir") in (None, ""):
        raise ValueError("Missing required setting: io.input_dir")
    if io_cfg.get("output_dir") in (None, ""):
        raise ValueError("Missing required setting: io.output_dir")

    image_cfg = settings.get("image", {})
    pixel = image_cfg.get("pixel_microns")
    if pixel is None or float(pixel) <= 0:
        raise ValueError("image.pixel_microns must be > 0")

    channels = image_cfg.get("channel_names")
    if not isinstance(channels, list) or len(channels) == 0:
        raise ValueError("image.channel_names must be a non-empty array")

    seg = settings.get("segmentation", {})
    method = seg.get("method")
    if method not in _ALLOWED_METHODS:
        raise ValueError(f"segmentation.method must be one of {sorted(_ALLOWED_METHODS)}")
    if int(seg.get("min_particle_size", 0)) < 0:
        raise ValueError("segmentation.min_particle_size must be >= 0")
    if int(seg.get("min_hole_size", 0)) < 0:
        raise ValueError("segmentation.min_hole_size must be >= 0")
    if int(seg.get("window_size", 1)) < 3:
        raise ValueError("segmentation.window_size must be >= 3")

    filtering = settings.get("filtering", {})
    if int(filtering.get("edge_distance", 0)) < 0:
        raise ValueError("filtering.edge_distance must be >= 0")
    _assert_numeric_range("filtering.area", filtering.get("area_min"), filtering.get("area_max"))
    _assert_numeric_range(
        "filtering.eccentricity",
        filtering.get("eccentricity_min"),
        filtering.get("eccentricity_max"),
    )
    _assert_numeric_range(
        "filtering.solidity",
        filtering.get("solidity_min"),
        filtering.get("solidity_max"),
    )
    _assert_numeric_range("filtering.length", filtering.get("length_min"), filtering.get("length_max"))
    _assert_numeric_range("filtering.width", filtering.get("width_min"), filtering.get("width_max"))
    _assert_numeric_range(
        "filtering.aspect_ratio",
        filtering.get("aspect_ratio_min"),
        filtering.get("aspect_ratio_max"),
    )

    centrality = settings.get("centrality", {})
    tie_breaker = centrality.get("tie_breaker")
    if tie_breaker not in _ALLOWED_TIE_BREAKERS:
        raise ValueError(f"centrality.tie_breaker must be one of {sorted(_ALLOWED_TIE_BREAKERS)}")
    if int(centrality.get("center_window_px", 0)) < 0:
        raise ValueError("centrality.center_window_px must be >= 0")

    runtime = settings.get("runtime", {})
    if int(runtime.get("workers", 0)) < 1:
        raise ValueError("runtime.workers must be >= 1")
    limit = runtime.get("limit")
    if limit is not None and int(limit) <= 0:
        raise ValueError("runtime.limit must be > 0 when provided")
    if not isinstance(runtime.get("log_level"), str):
        raise ValueError("runtime.log_level must be a string")


def resolve_effective_settings(
    *,
    settings_path: Path,
    import_app_settings: bool = False,
    user_settings_path: Path | None = None,
    app_config_path: Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    allow_missing_settings: bool = True,
    write_settings_if_missing: bool = False,
) -> dict[str, Any]:
    """
    Load and resolve effective settings.

    Precedence:
        defaults < app_settings < settings_json < cli_overrides
    """
    app_layer: dict[str, Any] = {}
    if import_app_settings:
        default_user_settings = Path("session_output/user_settings.json")
        default_app_config = Path("session_output/app_config.json")
        app_layer = load_app_compat_settings(
            user_settings_path or default_user_settings,
            app_config_path or default_app_config,
        )

    settings_layer: dict[str, Any] = {}
    if settings_path.exists():
        settings_layer = load_json_file(settings_path)
    elif not allow_missing_settings:
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    else:
        logger.warning(f"Settings file not found at {settings_path}; continuing with defaults + overrides.")

    merged = merge_settings(
        defaults=DEFAULT_SETTINGS,
        app_settings=app_layer,
        json_settings=settings_layer,
        cli_overrides=cli_overrides,
    )

    if write_settings_if_missing and not settings_path.exists():
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w") as handle:
            json.dump(merged, handle, indent=2)
        logger.info(f"Wrote generated settings file to {settings_path}")

    validate_settings(merged)
    return merged


def discover_images(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    """Discover TIFF images from input directory."""
    if recursive:
        candidates = sorted(input_dir.rglob(pattern))
    else:
        candidates = sorted(input_dir.glob(pattern))
    images = [
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]
    return images


def derive_output_paths(output_dir: Path, image_id: str) -> tuple[Path, Path]:
    """Derive output paths for a given image id."""
    mask_path = output_dir / "masks" / f"{image_id}_mask.tif"
    metadata_path = output_dir / "metadata" / f"{image_id}_meta.json"
    return mask_path, metadata_path


def build_ucid(source_path: str | Path, label: int, bbox: tuple[int, int, int, int], prefix: str) -> str:
    """Build deterministic UCID string from source path + label + bbox."""
    filename = Path(source_path).name
    hash_input = f"{filename}|{label}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def apply_ucids_to_metadata(metadata_path: Path, source_path: Path, prefix: str) -> None:
    """Append deterministic UCIDs to cells in metadata JSON."""
    metadata = load_json_file(metadata_path)
    cells = metadata.get("cells", [])
    if not isinstance(cells, list):
        return

    for cell in cells:
        if not isinstance(cell, dict):
            continue
        label = cell.get("label")
        bbox = cell.get("bbox")
        if label is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            label_int = int(label)
            bbox_tuple = tuple(int(x) for x in bbox)
        except (TypeError, ValueError):
            continue
        cell["ucid"] = build_ucid(source_path=source_path, label=label_int, bbox=bbox_tuple, prefix=prefix)

    metadata["ucids_enabled"] = True
    metadata["ucids_prefix"] = prefix
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)


def _default_channel_names(channel_count: int) -> list[str]:
    names: list[str] = []
    for idx in range(max(0, channel_count)):
        if idx < len(_DEFAULT_CHANNEL_NAMES):
            names.append(_DEFAULT_CHANNEL_NAMES[idx])
        else:
            names.append(f"Channel {idx + 1}")
    return names


def _normalize_channel_names(names: list[str] | None, channel_count: int) -> list[str]:
    defaults = _default_channel_names(channel_count)
    if not names:
        return defaults

    normalized = defaults[:]
    for idx in range(min(len(names), channel_count)):
        candidate = str(names[idx]).strip()
        if candidate:
            normalized[idx] = candidate
    return normalized


def _parse_finite_float(value: object) -> tuple[float | None, str]:
    if value is None:
        return None, "unavailable"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, "invalid"
    if not np.isfinite(parsed):
        return None, "invalid"
    return parsed, "ok"


def _compute_primary_failure_reasons(row: Mapping[str, Any], filter_cfg: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []

    if bool(filter_cfg.get("exclude_edge", True)) and bool(row.get("$touching_edge", 0)):
        reasons.append("Touching edge")

    area_raw = row.get("area")
    area, area_status = _parse_finite_float(area_raw)
    area_min = filter_cfg.get("area_min")
    area_max = filter_cfg.get("area_max")
    if area_status == "unavailable":
        reasons.append("Area unavailable")
    elif area_status == "invalid":
        reasons.append(f"Area invalid ({area_raw})")
    elif area is not None:
        if area_min is not None and area <= float(area_min):
            reasons.append(f"Area too small ({area:.0f} px^2 < {area_min})")
        elif area_max is not None and area >= float(area_max):
            reasons.append(f"Area too large ({area:.0f} px^2 > {area_max})")

    ecc_raw = row.get("eccentricity")
    ecc, ecc_status = _parse_finite_float(ecc_raw)
    ecc_min = filter_cfg.get("eccentricity_min")
    ecc_max = filter_cfg.get("eccentricity_max")
    if ecc_status == "unavailable":
        reasons.append("Eccentricity unavailable")
    elif ecc_status == "invalid":
        reasons.append(f"Eccentricity invalid ({ecc_raw})")
    elif ecc is not None:
        if ecc_min is not None and ecc < float(ecc_min):
            reasons.append(f"Eccentricity too low ({ecc:.2f})")
        elif ecc_max is not None and ecc > float(ecc_max):
            reasons.append(f"Eccentricity too high ({ecc:.2f})")

    sol_raw = row.get("solidity")
    sol, sol_status = _parse_finite_float(sol_raw)
    sol_min = filter_cfg.get("solidity_min")
    sol_max = filter_cfg.get("solidity_max")
    if sol_status == "unavailable":
        reasons.append("Solidity unavailable")
    elif sol_status == "invalid":
        reasons.append(f"Solidity invalid ({sol_raw})")
    elif sol is not None:
        if sol_min is not None and sol < float(sol_min):
            reasons.append(f"Solidity too low ({sol:.2f})")
        elif sol_max is not None and sol > float(sol_max):
            reasons.append(f"Solidity too high ({sol:.2f})")

    if not reasons and int(row.get("$include", 1)) == 0:
        reasons.append("Rejected by unmapped primary filter")

    return reasons


def _apply_additional_filters(metrics: Mapping[str, Any], filter_cfg: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []

    length = metrics.get("length_um")
    width = metrics.get("width_um")
    aspect = metrics.get("aspect_ratio")

    length_min = filter_cfg.get("length_min")
    length_max = filter_cfg.get("length_max")
    width_min = filter_cfg.get("width_min")
    width_max = filter_cfg.get("width_max")
    aspect_min = filter_cfg.get("aspect_ratio_min")
    aspect_max = filter_cfg.get("aspect_ratio_max")

    if length_min is not None:
        if length is None:
            reasons.append("Length unavailable")
        elif float(length) < float(length_min):
            reasons.append(f"Length too small ({float(length):.2f} um < {length_min})")
    if length_max is not None:
        if length is None:
            reasons.append("Length unavailable")
        elif float(length) > float(length_max):
            reasons.append(f"Length too large ({float(length):.2f} um > {length_max})")

    if width_min is not None:
        if width is None:
            reasons.append("Width unavailable")
        elif float(width) < float(width_min):
            reasons.append(f"Width too small ({float(width):.2f} um < {width_min})")
    if width_max is not None:
        if width is None:
            reasons.append("Width unavailable")
        elif float(width) > float(width_max):
            reasons.append(f"Width too large ({float(width):.2f} um > {width_max})")

    if aspect_min is not None:
        if aspect is None:
            reasons.append("Aspect ratio unavailable")
        elif float(aspect) < float(aspect_min):
            reasons.append(f"Aspect ratio too low ({float(aspect):.3f} < {aspect_min})")
    if aspect_max is not None:
        if aspect is None:
            reasons.append("Aspect ratio unavailable")
        elif float(aspect) > float(aspect_max):
            reasons.append(f"Aspect ratio too high ({float(aspect):.3f} > {aspect_max})")

    return reasons


def _apply_primary_filters(regionprops: Any, filter_cfg: Mapping[str, Any]) -> None:
    include_flags: list[int] = []
    for _, row in regionprops.iterrows():
        reasons = _compute_primary_failure_reasons(row, filter_cfg)
        include_flags.append(0 if reasons else 1)
    regionprops["$include"] = np.asarray(include_flags, dtype=int)


def _build_cells(
    *,
    regionprops: Any,
    filter_cfg: Mapping[str, Any],
    pixel_microns: float,
    channel_names: list[str],
    image_id: str,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    length_col = "length [\u00b5m]"
    width_col = "width_median [\u00b5m]"

    for label, row in regionprops.iterrows():
        is_included = bool(row.get("$include", 1) == 1)
        failure_reasons: list[str] = []
        if not is_included:
            failure_reasons = _compute_primary_failure_reasons(row, filter_cfg)

        area_raw = row.get("area", 0.0)
        try:
            area = float(area_raw)
            if not np.isfinite(area):
                area = 0.0
        except (TypeError, ValueError):
            area = 0.0

        metrics: dict[str, Any] = {
            "area": area,
            "area_px": area,
            "area_um2": area * (pixel_microns ** 2),
        }

        for col, key in [
            ("perimeter", "perimeter_px"),
            ("major_axis_length", "major_axis_px"),
            ("minor_axis_length", "minor_axis_px"),
            ("eccentricity", "eccentricity"),
            ("solidity", "solidity"),
            ("extent", "extent"),
            ("aspect_ratio", "aspect_ratio"),
        ]:
            value, status = _parse_finite_float(row.get(col))
            if status == "ok" and value is not None:
                metrics[key] = value

        length_value, length_status = _parse_finite_float(row.get(length_col))
        if length_status == "ok" and length_value is not None and length_value > 0:
            metrics["length_um"] = length_value

        width_value, width_status = _parse_finite_float(row.get(width_col))
        if width_status == "ok" and width_value is not None and width_value > 0:
            metrics["width_um"] = width_value

        sinuosity_value, sinuosity_status = _parse_finite_float(row.get("sinuosity"))
        if sinuosity_status == "ok" and sinuosity_value is not None and sinuosity_value > 0:
            metrics["sinuosity"] = sinuosity_value

        if is_included:
            extra_reasons = _apply_additional_filters(metrics, filter_cfg)
            if extra_reasons:
                is_included = False
                failure_reasons.extend(extra_reasons)

        intensity: dict[str, float] = {}
        for channel in channel_names:
            mean_col = f"{channel}_mean"
            value, status = _parse_finite_float(row.get(mean_col))
            if status == "ok" and value is not None:
                intensity[channel] = value
        if intensity:
            metrics["intensity"] = intensity

        cell = {
            "label": int(label),
            "cell_id": str(row.get("$Cell_id", f"{image_id}_{int(label)}")),
            "included": is_included,
            "override": None,
            "note": None,
            "failure_reasons": failure_reasons,
            "bbox": [
                int(row.get("$opt-x1", 0)),
                int(row.get("$opt-y1", 0)),
                int(row.get("$opt-x2", 0)),
                int(row.get("$opt-y2", 0)),
            ],
            "centroid": [
                float(row.get("$centroid-0", 0.0)),
                float(row.get("$centroid-1", 0.0)),
            ],
            "touching_edge": bool(row.get("$touching_edge", 0)),
            "metrics": metrics,
            "is_central": False,
            "distance_to_center_px": None,
        }
        cells.append(cell)

    return cells


def _apply_central_selection(
    labeled_mask: np.ndarray,
    cells: list[dict[str, Any]],
    central_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    center_window_px = int(central_cfg.get("center_window_px", 20))
    centrality = {
        "central_label": None,
        "central_distance_px": None,
        "center_window_px": center_window_px,
        "centrality_pass": False,
        "centrality_reason": None,
    }

    if labeled_mask.max() == 0:
        centrality["centrality_reason"] = "No objects detected"
        return np.zeros_like(labeled_mask), 0, 0, centrality

    full_mask = labeled_mask.copy()
    height, width = full_mask.shape
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    x_min = max(0, int(cx - center_window_px))
    x_max = min(width, int(cx + center_window_px + 1))
    y_min = max(0, int(cy - center_window_px))
    y_max = min(height, int(cy + center_window_px + 1))

    labels_in_window = set(np.unique(full_mask[y_min:y_max, x_min:x_max])) - {0}
    if not labels_in_window:
        centrality["centrality_reason"] = "No object intersects central window"
        return np.zeros_like(full_mask), 0, 0, centrality

    tie_breaker = str(central_cfg.get("tie_breaker", "area"))
    candidates: list[tuple[dict[str, Any], float, float]] = []
    for cell in cells:
        if int(cell["label"]) not in labels_in_window:
            continue
        centroid = cell.get("centroid", [0.0, 0.0])
        dist = float(((float(centroid[0]) - cy) ** 2 + (float(centroid[1]) - cx) ** 2) ** 0.5)
        cell["distance_to_center_px"] = dist

        tie_value_raw = cell.get("metrics", {}).get(tie_breaker)
        if tie_value_raw is None:
            tie_value_raw = cell.get("metrics", {}).get("area_px", 0.0)
        tie_value, tie_status = _parse_finite_float(tie_value_raw)
        if tie_status != "ok" or tie_value is None:
            tie_value = 0.0
        candidates.append((cell, dist, tie_value))

    if not candidates:
        centrality["centrality_reason"] = "No cell data for labels in central window"
        return np.zeros_like(full_mask), 0, 0, centrality

    candidates.sort(key=lambda item: (item[1], -item[2]))
    central_cell, central_dist, _ = candidates[0]
    central_label = int(central_cell["label"])
    central_cell["is_central"] = True

    centrality["central_label"] = central_label
    centrality["central_distance_px"] = central_dist

    for cell in cells:
        if int(cell["label"]) == central_label:
            continue
        cell["included"] = False
        reasons = cell.setdefault("failure_reasons", [])
        reasons.append("Non-central cell")

    fail_if_filtered = bool(central_cfg.get("fail_if_center_cell_filtered", True))
    if fail_if_filtered and not bool(central_cell.get("included", False)):
        reasons = central_cell.get("failure_reasons", [])
        reason_text = ", ".join(reasons) if reasons else "Unknown"
        centrality["centrality_reason"] = f"Central object failed filters: {reason_text}"
        new_mask = np.zeros_like(full_mask, dtype=np.uint16)
        new_mask[full_mask == central_label] = central_label
        return new_mask, 1, 0, centrality

    centrality["centrality_pass"] = True
    centrality["centrality_reason"] = "Central object selected"
    new_mask = np.zeros_like(full_mask, dtype=np.uint16)
    new_mask[full_mask == central_label] = central_label
    return new_mask, 1, 1, centrality


def _save_mask_tiff(mask: np.ndarray, path: Path) -> None:
    mask_16 = mask.astype(np.uint16)
    n_labels = int(mask.max())
    imagej_metadata = {
        "Labels": [f"{n_labels} labels"],
        "Info": f"labels={n_labels}",
    }
    tifffile.imwrite(str(path), mask_16, imagej=True, metadata=imagej_metadata)


def _clean_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _build_metadata_payload(
    *,
    image_id: str,
    source_path: Path,
    image_shape: tuple[int, ...],
    pixel_microns: float,
    seg_cfg: Mapping[str, Any],
    filter_cfg: Mapping[str, Any],
    central_cfg: Mapping[str, Any],
    n_cells_total: int,
    n_cells_filtered: int,
    n_touching_edge: int,
    cells: list[dict[str, Any]],
    centrality: Mapping[str, Any],
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "id": image_id,
        "image_id": image_id,
        "source_path": str(source_path),
        "status": "unreviewed",
        "timestamp": datetime.now().isoformat(),
        "image_shape": [int(x) for x in image_shape],
        "pixel_microns": float(pixel_microns),
        "segmentation_params": {
            "method": str(seg_cfg["method"]),
            "min_particle_size": int(seg_cfg["min_particle_size"]),
            "min_hole_size": int(seg_cfg["min_hole_size"]),
            "dark_foreground": bool(seg_cfg["dark_foreground"]),
            "window_size": int(seg_cfg["window_size"]),
            "k": float(seg_cfg["k"]),
            "unet_checkpoint": None,
            "unet_cell_prob_min": 0.6,
            "unet_cell_margin": 0.1,
        },
        "filter_params": {
            "exclude_edge": bool(filter_cfg["exclude_edge"]),
            "edge_distance": int(filter_cfg["edge_distance"]),
            "area_min": float(filter_cfg["area_min"]),
            "area_max": float(filter_cfg["area_max"]),
            "eccentricity_min": float(filter_cfg["eccentricity_min"]),
            "eccentricity_max": float(filter_cfg["eccentricity_max"]),
            "solidity_min": float(filter_cfg["solidity_min"]),
            "solidity_max": float(filter_cfg["solidity_max"]),
            "length_min": filter_cfg["length_min"],
            "length_max": filter_cfg["length_max"],
            "width_min": filter_cfg["width_min"],
            "width_max": filter_cfg["width_max"],
            "aspect_ratio_min": filter_cfg["aspect_ratio_min"],
            "aspect_ratio_max": filter_cfg["aspect_ratio_max"],
        },
        "centrality_params": {
            "enabled": bool(central_cfg["enabled"]),
            "center_window_px": int(central_cfg["center_window_px"]),
            "tie_breaker": str(central_cfg["tie_breaker"]),
            "fail_if_center_cell_filtered": bool(central_cfg["fail_if_center_cell_filtered"]),
        },
        "stats": {
            "n_cells_total": int(n_cells_total),
            "n_cells_filtered": int(n_cells_filtered),
            "n_touching_edge": int(n_touching_edge),
        },
        "centrality": {
            "central_label": centrality.get("central_label"),
            "central_distance_px": centrality.get("central_distance_px"),
            "center_window_px": centrality.get("center_window_px"),
            "centrality_pass": bool(centrality.get("centrality_pass", False)),
            "centrality_reason": centrality.get("centrality_reason"),
        },
        "cells": cells,
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    return _clean_for_json(metadata)


def _segment_and_save(
    image_path: Path,
    output_dir: Path,
    settings: Mapping[str, Any],
) -> tuple[int, int, Path, Path]:
    """
    Run segmentation and save outputs.

    Returns:
        n_cells_total, n_cells_filtered, mask_path, metadata_path
    """
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from momia2.core.patch import Patch

    image_cfg = settings["image"]
    seg_cfg = settings["segmentation"]
    filter_cfg = settings["filtering"]
    central_cfg = settings["centrality"]
    metadata_cfg = settings["metadata"]

    pixel_microns = float(image_cfg["pixel_microns"])
    img_stack = tifffile.imread(str(image_path))
    if img_stack.ndim == 2:
        img_stack = img_stack[np.newaxis, ...]
    if img_stack.ndim != 3:
        raise ValueError(f"Expected (C, H, W) stack, got {img_stack.shape}")

    channel_names = _normalize_channel_names(image_cfg["channel_names"], img_stack.shape[0])
    ref_channel = channel_names[0]
    img_dict = {name: img_stack[idx] for idx, name in enumerate(channel_names)}

    patch = Patch(
        image_dict=img_dict,
        image_id=image_path.stem,
        ref_channel=ref_channel,
    )
    patch.pixel_microns = pixel_microns
    patch.correct_xy_drift(reference_channel=ref_channel, max_drift=8)

    method_kwargs: dict[str, Any] = {"dark_foreground": bool(seg_cfg["dark_foreground"])}
    method_name = str(seg_cfg["method"])
    if method_name in {"sauvola", "local"}:
        method_kwargs["window_size"] = int(seg_cfg["window_size"])
    if method_name == "sauvola":
        method_kwargs["k"] = float(seg_cfg["k"])

    patch.generate_mask(
        method=_MOMIA_METHOD_MAP[method_name],
        min_particle_size=int(seg_cfg["min_particle_size"]),
        min_hole_size=int(seg_cfg["min_hole_size"]),
        **method_kwargs,
    )
    patch.label_mask()

    labeled_mask = patch.labeled_mask
    n_cells_total = 0
    n_cells_filtered = 0
    n_touching_edge = 0
    cells: list[dict[str, Any]] = []
    centrality = {
        "central_label": None,
        "central_distance_px": None,
        "center_window_px": int(central_cfg["center_window_px"]) if bool(central_cfg["enabled"]) else None,
        "centrality_pass": False,
        "centrality_reason": None,
    }

    if labeled_mask.max() != 0:
        patch.locate_particles(use_intensity=True, edge_width=int(filter_cfg["edge_distance"]))
        _apply_primary_filters(patch.regionprops, filter_cfg)
        n_cells_total = int(len(patch.regionprops))
        n_touching_edge = int(patch.regionprops["$touching_edge"].sum()) if "$touching_edge" in patch.regionprops else 0

        if n_cells_total > 0:
            try:
                patch.find_outline()
                patch.extract_midlines(method="zhang", filtered_only=False)
                patch.get_morphology_stats()
                patch.get_intensity_stats()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Feature extraction failed for {image_path.stem}: {exc}. Continuing with base regionprops.")

        cells = _build_cells(
            regionprops=patch.regionprops,
            filter_cfg=filter_cfg,
            pixel_microns=pixel_microns,
            channel_names=channel_names,
            image_id=image_path.stem,
        )
        n_cells_filtered = sum(1 for cell in cells if bool(cell["included"]))

    if bool(central_cfg["enabled"]):
        labeled_mask, n_cells_total, n_cells_filtered, centrality = _apply_central_selection(
            labeled_mask=labeled_mask,
            cells=cells,
            central_cfg=central_cfg,
        )

    extra_metadata: dict[str, Any] = {
        "pipeline": "momia2_batch_segmentation",
    }
    if bool(metadata_cfg.get("embed_effective_settings", True)):
        extra_metadata["effective_settings"] = {
            "image": copy.deepcopy(dict(settings["image"])),
            "segmentation": copy.deepcopy(dict(settings["segmentation"])),
            "filtering": copy.deepcopy(dict(settings["filtering"])),
            "centrality": copy.deepcopy(dict(settings["centrality"])),
            "metadata": copy.deepcopy(dict(settings["metadata"])),
        }

    mask_path, metadata_path = derive_output_paths(output_dir, image_path.stem)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _save_mask_tiff(labeled_mask, mask_path)

    metadata_payload = _build_metadata_payload(
        image_id=image_path.stem,
        source_path=image_path,
        image_shape=tuple(int(x) for x in img_stack.shape),
        pixel_microns=pixel_microns,
        seg_cfg=seg_cfg,
        filter_cfg=filter_cfg,
        central_cfg=central_cfg,
        n_cells_total=n_cells_total,
        n_cells_filtered=n_cells_filtered,
        n_touching_edge=n_touching_edge,
        cells=cells,
        centrality=centrality,
        extra_metadata=extra_metadata,
    )
    with open(metadata_path, "w") as handle:
        json.dump(metadata_payload, handle, indent=2)

    if bool(metadata_cfg.get("emit_ucid", True)):
        apply_ucids_to_metadata(
            metadata_path=metadata_path,
            source_path=image_path,
            prefix=str(metadata_cfg.get("ucid_prefix", "m2_")),
        )

    return n_cells_total, n_cells_filtered, mask_path, metadata_path


def process_single_image(image_path: Path, settings: Mapping[str, Any]) -> ImageProcessResult:
    """Process one image and return run status."""
    io_cfg = settings["io"]
    runtime_cfg = settings["runtime"]
    output_dir = Path(io_cfg["output_dir"])

    image_id = image_path.stem
    mask_path, metadata_path = derive_output_paths(output_dir, image_id)

    force = bool(runtime_cfg["force"])
    skip_existing = bool(runtime_cfg["skip_existing"])
    dry_run = bool(runtime_cfg["dry_run"])
    existing_outputs = mask_path.exists() and metadata_path.exists()

    if not force and skip_existing and existing_outputs:
        return ImageProcessResult(
            image_id=image_id,
            source_path=str(image_path),
            status="skipped",
            mask_path=str(mask_path),
            metadata_path=str(metadata_path),
        )

    if dry_run:
        return ImageProcessResult(
            image_id=image_id,
            source_path=str(image_path),
            status="dry_run",
            mask_path=str(mask_path),
            metadata_path=str(metadata_path),
        )

    try:
        n_cells_total, n_cells_filtered, out_mask_path, out_meta_path = _segment_and_save(
            image_path=image_path,
            output_dir=output_dir,
            settings=settings,
        )
        return ImageProcessResult(
            image_id=image_id,
            source_path=str(image_path),
            status="success",
            n_cells_total=n_cells_total,
            n_cells_filtered=n_cells_filtered,
            mask_path=str(out_mask_path),
            metadata_path=str(out_meta_path),
        )
    except Exception as exc:  # noqa: BLE001
        return ImageProcessResult(
            image_id=image_id,
            source_path=str(image_path),
            status="failed",
            error=str(exc),
        )


def _worker_process_single(image_path: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Process one image in a worker process."""
    result = process_single_image(Path(image_path), settings)
    return asdict(result)


def run_batch(settings: Mapping[str, Any], settings_path: Path | None = None) -> BatchRunSummary:
    """Run batch MOMIA2 segmentation for all matching images."""
    io_cfg = settings["io"]
    runtime_cfg = settings["runtime"]
    input_dir = Path(io_cfg["input_dir"])
    output_dir = Path(io_cfg["output_dir"])
    pattern = str(io_cfg["pattern"])
    recursive = bool(io_cfg["recursive"])
    workers = int(runtime_cfg["workers"])
    limit = runtime_cfg["limit"]
    fail_fast = bool(runtime_cfg["fail_fast"])
    dry_run = bool(runtime_cfg["dry_run"])

    images = discover_images(input_dir=input_dir, pattern=pattern, recursive=recursive)
    total_candidates = len(images)
    if limit is not None:
        images = images[: int(limit)]

    if not dry_run:
        (output_dir / "masks").mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().isoformat()
    results: list[ImageProcessResult] = []

    if workers > 1 and fail_fast:
        logger.warning("runtime.fail_fast is enabled; running sequentially to support deterministic early stop.")
        workers = 1

    if workers <= 1 or dry_run:
        for image_path in images:
            result = process_single_image(image_path=image_path, settings=settings)
            results.append(result)
            if result.status == "failed":
                logger.error(f"{result.image_id}: {result.error}")
                if fail_fast:
                    break
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker_process_single, str(path), settings): path
                for path in images
            }
            for future in as_completed(futures):
                payload = future.result()
                results.append(ImageProcessResult(**payload))

    results.sort(key=lambda item: item.image_id)
    succeeded = sum(1 for item in results if item.status == "success")
    failed = sum(1 for item in results if item.status == "failed")
    skipped = sum(1 for item in results if item.status == "skipped")
    processed = len(results)

    summary = BatchRunSummary(
        started_at=started_at,
        finished_at=datetime.now().isoformat(),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        pattern=pattern,
        recursive=recursive,
        total_candidates=total_candidates,
        limit_applied=int(limit) if limit is not None else None,
        processed=processed,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        dry_run=dry_run,
        settings_path=str(settings_path) if settings_path else None,
        results=results,
    )

    if not dry_run:
        write_run_artifacts(output_dir=output_dir, settings=settings, summary=summary)

    return summary


def write_run_artifacts(
    output_dir: Path,
    settings: Mapping[str, Any],
    summary: BatchRunSummary,
) -> None:
    """Write effective settings and run summary files."""
    settings_path = output_dir / "effective_settings.json"
    summary_path = output_dir / "run_summary.json"

    with open(settings_path, "w") as handle:
        json.dump(dict(settings), handle, indent=2)

    with open(summary_path, "w") as handle:
        json.dump(
            {
                **asdict(summary),
                "results": [asdict(item) for item in summary.results],
            },
            handle,
            indent=2,
        )
