"""Feature-extraction stage for the image-segmentation pipeline.

This module produces deciphaer-compatible per-cell feature tables from MOMIA
masks + crops. It runs the vendored extractors (regionprops, centerline,
intensity, fluorescence) on the FULL source image + FULL mask for each
cell so the features match the original deciphaer pipeline's output exactly.

Channel handling is driven by :mod:`channels`: every detected non-phase
channel gets the common intensity/profile/fluorescence-common feature set,
and channels matching a registered plugin (AF405 → HADA, Bod493 → puncta)
get bespoke extras layered on top. Adding a new bespoke panel is a single
edit to ``CHANNEL_PLUGINS`` in :mod:`channels`.

Public entry point:
    :func:`run_extract`
"""

from __future__ import annotations

import hashlib
import json
import logging
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from deciphaer_image_segmentation.cell_utils import filter_unet, join_crops
from deciphaer_image_segmentation.channels import (
    KNOWN_CHANNELS,
    PluginContext,
    discover_sibling_channels,
    empty_fluor_common_features,
    empty_intensity_features,
    empty_profile_features,
    feature_columns_for_channels,
    parse_channel_token,
    plugins_for_channel,
)

_log = logging.getLogger(__name__)


def _empty_channel_fluor_block(channel: str) -> dict[str, float]:
    """All fluorescence features (common + any matching plugin extras) as NaN."""
    feats = empty_fluor_common_features(channel)
    for plugin in plugins_for_channel(channel):
        feats.update(plugin.empty_features(channel))
    return feats


# ---------------------------------------------------------------------------
# I/O helpers (mirror crops.py conventions)
# ---------------------------------------------------------------------------
def _read_2d(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF at {path}, got {arr.shape}")
    return arr


def _base_from_phase(
    source: Path,
    phase_channel: str,
    *,
    known_channels: tuple[str, ...] = KNOWN_CHANNELS,
) -> str:
    _, base = parse_channel_token(
        source.stem, known_channels=known_channels, phase_channel=phase_channel,
    )
    return base


def _resolve_channel_paths_for_source(
    source: Path,
    *,
    channel_names: tuple[str, ...],
    phase_channel: str,
    known_channels: tuple[str, ...] = KNOWN_CHANNELS,
) -> dict[str, Path]:
    """Resolve every requested channel's sibling path next to ``source``.

    Walks ``source.parent`` once via :func:`discover_sibling_channels` and
    falls back to legacy strict-suffix / name-substitution lookups for any
    channel that didn't appear in the discovered siblings. Missing channels
    are simply absent from the return dict (the caller fills with NaNs).
    """
    out: dict[str, Path] = {}
    siblings = discover_sibling_channels(
        source, phase_channel=phase_channel, known_channels=known_channels,
    )
    for channel in channel_names:
        if channel == phase_channel:
            if source.exists():
                out[channel] = source
            continue
        if channel in siblings:
            out[channel] = siblings[channel]
            continue
        legacy = _resolve_channel_path_legacy(source, channel=channel, phase_channel=phase_channel)
        if legacy is not None:
            out[channel] = legacy
    return out


def _resolve_channel_path_legacy(source: Path, *, channel: str, phase_channel: str) -> Path | None:
    """Fallback path lookup for the strict ``<base>_<channel>.tif`` convention."""
    phase_suffix = f"_{phase_channel}"
    if source.stem.endswith(phase_suffix):
        base = source.stem[: -len(phase_suffix)]
        candidate = source.with_name(f"{base}_{channel}{source.suffix}")
        if candidate.exists():
            return candidate
    candidate = source.with_name(source.name.replace(phase_channel, channel))
    if candidate.exists():
        return candidate
    return None


# ---------------------------------------------------------------------------
# Cell hashing — mirrors crops.py and deciphaer's hashing module.
# ---------------------------------------------------------------------------
def _cell_hash_digest(source_path: Path, label: int, bbox: tuple[int, int, int, int]) -> str:
    """Same algorithm as deciphaer.processing.hashing.cell_hash.generate_cell_hash."""
    filename = Path(source_path).name
    raw = f"{filename}|{label}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------
def _bbox_from_row(row: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return ``(min_row, min_col, max_row, max_col)`` from a cell-table row.

    MOMIA's columns ``bbox_x1/y1/x2/y2`` are misleadingly named: ``x1`` is
    actually ``min_row`` and ``y1`` is actually ``min_col``. ``optimize_bbox``
    in MOMIA reads skimage regionprops' ``bbox-0`` (=min_row) into ``$opt-x1``,
    so the values are correct row/col coords with a swapped letter label.
    We use the MOMIA columns as the source of truth and interpret them
    correctly.

    ``bbox_min_row``/``bbox_min_col`` from a crops manifest are themselves
    swapped (crops.py copies MOMIA's mislabeled fields verbatim), so we
    deliberately prefer the raw MOMIA columns when both are present.
    """
    def _safe_int(value: Any, fallback: int) -> int:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return fallback
        if not np.isfinite(f):
            return fallback
        return int(f)

    if "bbox_x1" in row and pd.notna(row.get("bbox_x1")):
        x1 = _safe_int(row.get("bbox_x1"), 0)  # = min_row
        y1 = _safe_int(row.get("bbox_y1"), 0)  # = min_col
        x2 = _safe_int(row.get("bbox_x2"), x1 + 1)  # = max_row
        y2 = _safe_int(row.get("bbox_y2"), y1 + 1)  # = max_col
        return x1, y1, x2, y2
    if "bbox_min_row" in row and pd.notna(row.get("bbox_min_row")):
        return (
            _safe_int(row.get("bbox_min_row"), 0),
            _safe_int(row.get("bbox_min_col"), 0),
            _safe_int(row.get("bbox_max_row"), 0),
            _safe_int(row.get("bbox_max_col"), 0),
        )
    return 0, 0, 1, 1


# ---------------------------------------------------------------------------
# Per-channel fluorescence pass
# ---------------------------------------------------------------------------
def _extract_channel_fluor(
    *,
    channel: str,
    ch_img: np.ndarray,
    binary_mask: np.ndarray,
    shell: np.ndarray | None,
    core: np.ndarray | None,
    bg_ring: np.ndarray | None,
    midline: np.ndarray | None,
    cell_id: str,
    fluor_cfg: Any,
    puncta_records: list[dict[str, Any]],
) -> tuple[dict[str, float], np.ndarray | None]:
    """Compute the shared fluorescence features for one channel + any plugin extras.

    Returns ``(features, profile)`` where ``profile`` is the
    background-subtracted, reoriented midline profile (or ``None``).
    Mutates ``puncta_records`` only via plugins that need to write per-event
    rows (e.g. lipid bodies).
    """
    from fluorescence.background import background_stats, shell_core_stats
    from fluorescence.profiles import (
        compute_midline_profile,
        profile_scalar_summaries,
        reorient_profile,
    )

    feats: dict[str, float] = {}
    try:
        bg = background_stats(ch_img, bg_ring) if bg_ring is not None else {
            "bg_mean": np.nan, "bg_median": np.nan, "bg_std": np.nan,
        }
        feats[f"{channel}_bg_mean"] = bg.get("bg_mean", np.nan)
        feats[f"{channel}_bg_median"] = bg.get("bg_median", np.nan)
        feats[f"{channel}_bg_std"] = bg.get("bg_std", np.nan)

        bg_val = bg.get("bg_mean", 0.0)
        if not isinstance(bg_val, (int, float)) or np.isnan(bg_val):
            bg_val = 0.0
        bg_val = float(bg_val)
        cell_pixels = ch_img[binary_mask.astype(bool)]
        feats[f"{channel}_mean_bgsub"] = (
            float(np.mean(cell_pixels)) - bg_val if cell_pixels.size > 0 else np.nan
        )

        if shell is not None and core is not None:
            sc = shell_core_stats(ch_img, shell, core)
        else:
            sc = {"shell_mean": np.nan, "core_mean": np.nan, "shell_core_ratio": np.nan}
        feats[f"{channel}_shell_mean"] = sc.get("shell_mean", np.nan)
        feats[f"{channel}_core_mean"] = sc.get("core_mean", np.nan)
        feats[f"{channel}_shell_core_ratio"] = sc.get("shell_core_ratio", np.nan)

        profile = compute_midline_profile(
            midline, ch_img,
            width=fluor_cfg.strip_width, mode=fluor_cfg.strip_mode,
            n_points=fluor_cfg.n_profile_points,
        )
        if profile is not None:
            profile, _flipped = reorient_profile(profile)

        scalars = profile_scalar_summaries(profile)
        for k, v in scalars.items():
            feats[f"{channel}_{k}"] = v

        bg_sub_profile = (profile - bg_val) if profile is not None else None
        matched = plugins_for_channel(channel)
        if matched:
            # Plugins record their per-event sidecar rows tagged with channel
            # only; the caller stamps the cell_id once on flush.
            local_puncta: list[dict[str, Any]] = []
            ctx = PluginContext(
                channel=channel,
                image=ch_img,
                binary_mask=binary_mask,
                cell_pixels=cell_pixels,
                midline=midline,
                profile=bg_sub_profile,
                core_mask=core,
                shell_mask=shell,
                bg_value=bg_val,
                fluor_cfg=fluor_cfg,
                puncta_records_out=local_puncta,
            )
            for plugin in matched:
                try:
                    feats.update(plugin.extract(ctx))
                except Exception as exc:
                    _log.warning(
                        "extract: plugin %s failed for %s ch=%s: %s",
                        plugin.name, cell_id, channel, exc,
                    )
                    feats.update(plugin.empty_features(channel))
            for rec in local_puncta:
                rec["cell_id"] = cell_id
                puncta_records.append(rec)
        return feats, profile
    except Exception as exc:
        _log.warning("extract: fluorescence failed for %s ch=%s: %s", cell_id, channel, exc)
        return _empty_channel_fluor_block(channel), None


# ---------------------------------------------------------------------------
# Worker-side feature extraction (per-image)
# ---------------------------------------------------------------------------
def _extract_image_worker(args: tuple[Any, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Process every cell of one source image.  Returns (features, profile_records, puncta_records, image_info)."""
    (
        image_id,
        rows,
        masks_dir,
        pixel_microns,
        channel_names,
        phase_channel,
        fluorescence_config,
        known_channels,
    ) = args

    from deciphaer_processing.features.regionprops import extract_regionprops
    from deciphaer_processing.features.centerline import extract_centerline_features, compute_midline
    from deciphaer_processing.features.intensity import extract_intensity_features
    from fluorescence import FluorescenceConfig
    from fluorescence.background import (
        compute_background_ring,
        background_stats,
        compute_shell_core,
        shell_core_stats,
    )
    from fluorescence.profiles import (
        compute_midline_profile,
        profile_scalar_summaries,
        reorient_profile,
    )

    fluor_cfg = FluorescenceConfig(**(fluorescence_config or {}))

    feature_records: list[dict[str, Any]] = []
    profile_records: list[dict[str, Any]] = []
    puncta_records: list[dict[str, Any]] = []
    image_info: dict[str, Any] = {"image_id": image_id, "channels_loaded": [], "channels_missing": [], "n_cells": 0}

    if not rows:
        return feature_records, profile_records, puncta_records, image_info

    source = Path(str(rows[0]["source_path"])).resolve()
    mask_path = Path(masks_dir) / f"{image_id}_mask.tif"
    try:
        mask = _read_2d(mask_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        _log.warning("extract: mask unreadable for image_id=%s (%s) — skipping %d cells",
                     image_id, exc, len(rows))
        return feature_records, profile_records, puncta_records, image_info

    # Resolve + load channels for this image. Missing channels become NaN later.
    # One parent-dir scan resolves every requested channel.
    channel_arrays: dict[str, np.ndarray] = {}
    channel_paths: dict[str, Path] = {}
    resolved = _resolve_channel_paths_for_source(
        source,
        channel_names=tuple(channel_names),
        phase_channel=phase_channel,
        known_channels=tuple(known_channels),
    )
    for channel in channel_names:
        cpath = resolved.get(channel)
        if cpath is None:
            image_info["channels_missing"].append(channel)
            continue
        try:
            channel_arrays[channel] = _read_2d(cpath)
            channel_paths[channel] = cpath
        except (FileNotFoundError, ValueError, OSError) as exc:
            _log.warning("extract: channel %s unreadable at %s (%s)", channel, cpath, exc)
            image_info["channels_missing"].append(channel)
    image_info["channels_loaded"] = list(channel_arrays.keys())

    if phase_channel not in channel_arrays:
        # Without phase we cannot run regionprops with the right intensity image.
        # Fall back to an all-zero intensity image of the mask's shape.
        H, W = mask.shape
        channel_arrays[phase_channel] = np.zeros((H, W), dtype=np.float32)

    H, W = mask.shape
    # Build the (C, H, W) stack in `channel_names` order, filling missing channels with zeros.
    stacked = np.stack([
        channel_arrays.get(ch, np.zeros((H, W), dtype=np.float32))
        for ch in channel_names
    ], axis=0)

    image_info["n_cells"] = len(rows)
    H_full, W_full = mask.shape
    # Padding around each cell's bbox before cropping. The bg-ring dilation
    # plus a small buffer for spline smoothing is the largest local-context
    # any extractor needs; 24 px is comfortably above what the deciphaer
    # defaults consume.
    crop_pad = 24

    for row in rows:
        try:
            label = int(row["label"])
        except (KeyError, TypeError, ValueError):
            continue
        bbox = _bbox_from_row(row)
        cell_hash = str(row.get("cell_hash", "")) or _cell_hash_digest(source, label, bbox)
        # Strip any "sha256:" prefix the v2 manifest uses.
        cell_hash_raw = cell_hash.split(":", 1)[1] if cell_hash.startswith("sha256:") else cell_hash
        cell_id = str(row.get("Cell_ID", "")) or f"sc_{cell_hash_raw[:16]}"

        # Crop mask + channel stack to a padded bbox around this cell. All
        # downstream extractors then operate on a tiny patch (typically ~100²
        # vs 2304²), which is the difference between "8h slurm timeout" and
        # "minutes" on a 40k-cell run. Translation-invariant features
        # (areas, intensities, midline lengths, fluorescence stats) come out
        # identical; coords (centroid, bbox in regionprops output) are in
        # crop space and get re-translated to global below.
        y0 = max(0, bbox[0] - crop_pad)
        x0 = max(0, bbox[1] - crop_pad)
        y1 = min(H_full, bbox[2] + crop_pad)
        x1 = min(W_full, bbox[3] + crop_pad)
        if y1 <= y0 or x1 <= x0:
            # Degenerate bbox — skip this cell rather than crash.
            _log.warning("extract: degenerate bbox for %s (label=%d): %s — skipping",
                         cell_id, label, bbox)
            continue
        mask_crop = mask[y0:y1, x0:x1]
        stacked_crop = stacked[:, y0:y1, x0:x1]
        phase_crop = stacked_crop[0]

        try:
            rprops = extract_regionprops(
                mask=mask_crop,
                intensity_image=phase_crop,
                pixel_microns=pixel_microns,
                label=label,
            )
        except Exception as exc:
            _log.warning("extract: regionprops failed for %s (label=%d): %s", cell_id, label, exc)
            rprops = {}

        # Translate centroid + bbox features back to global image coordinates.
        # Other regionprops outputs (areas, axes, hu_moments, etc.) are
        # translation-invariant, so they don't need adjustment.
        # `extract_regionprops` returns NaN-filled dicts for masks where the
        # label is missing (e.g. crop padding clipped the cell); guard
        # against those before doing the int+offset.
        if rprops:
            for key, off in (("centroid_row", y0), ("centroid_col", x0)):
                v = rprops.get(key)
                if v is not None and pd.notna(v):
                    rprops[key] = float(v) + off
            for key, off in (
                ("bbox_min_row", y0), ("bbox_max_row", y0),
                ("bbox_min_col", x0), ("bbox_max_col", x0),
            ):
                v = rprops.get(key)
                if v is not None and pd.notna(v):
                    rprops[key] = int(v) + off

        try:
            cprops = extract_centerline_features(mask_crop, pixel_microns, label)
        except Exception as exc:
            _log.warning("extract: centerline failed for %s (label=%d): %s", cell_id, label, exc)
            cprops = {}

        try:
            midline = compute_midline(mask_crop, pixel_microns, label)
        except Exception as exc:
            _log.warning("extract: midline failed for %s (label=%d): %s", cell_id, label, exc)
            midline = None

        try:
            iprops = extract_intensity_features(
                mask=mask_crop,
                image=stacked_crop,
                channel_names=list(channel_names),
                midline=midline,
                label=label,
            )
        except Exception as exc:
            _log.warning("extract: intensity failed for %s (label=%d): %s", cell_id, label, exc)
            iprops = {}

        # For channels the source did NOT actually have, overwrite with NaN.
        for ch in image_info["channels_missing"]:
            iprops.update(empty_intensity_features(ch))
            iprops.update(empty_profile_features(ch))

        # Fluorescence pass per non-phase channel that was actually loaded.
        # Operates on the cropped binary mask + cropped channel image.
        binary_mask = (mask_crop == label).astype(np.uint8) if label else (mask_crop > 0).astype(np.uint8)
        try:
            shell, core = compute_shell_core(binary_mask, fluor_cfg.shell_erosion_radius)
            bg_ring = compute_background_ring(
                binary_mask,
                dilation_r=fluor_cfg.bg_ring_dilation,
                erosion_r=fluor_cfg.bg_ring_erosion,
            )
        except Exception as exc:
            _log.warning("extract: shell/core/bg-ring failed for %s: %s", cell_id, exc)
            shell = core = bg_ring = None

        fluor_feats: dict[str, Any] = {}
        for ch_idx, ch in enumerate(channel_names):
            if ch == phase_channel:
                continue
            if ch not in channel_arrays:
                fluor_feats.update(_empty_channel_fluor_block(ch))
                continue
            ch_img = stacked_crop[ch_idx].astype(np.float64)
            ch_feats, ch_profile = _extract_channel_fluor(
                channel=ch,
                ch_img=ch_img,
                binary_mask=binary_mask,
                shell=shell,
                core=core,
                bg_ring=bg_ring,
                midline=midline,
                cell_id=cell_id,
                fluor_cfg=fluor_cfg,
                puncta_records=puncta_records,
            )
            fluor_feats.update(ch_feats)
            if ch_profile is not None:
                profile_records.append({
                    "cell_id": cell_id,
                    "channel": ch,
                    "profile_kind": f"midline_strip_{fluor_cfg.strip_mode}",
                    "n_points": fluor_cfg.n_profile_points,
                    "profile": ch_profile.tolist(),
                })

        all_features: dict[str, Any] = {}
        all_features.update(rprops or {})
        all_features.update(cprops or {})
        all_features.update(iprops or {})
        all_features.update(fluor_feats)

        feature_records.append({
            "Cell_ID": cell_id,
            "cell_hash": cell_hash_raw,
            "image_id": image_id,
            "label": label,
            "source_path": str(source),
            "source_filename": source.name,
            "mask_path": str(mask_path),
            "bbox_min_row": bbox[0],
            "bbox_min_col": bbox[1],
            "bbox_max_row": bbox[2],
            "bbox_max_col": bbox[3],
            "crop_path": row.get("crop_path", ""),
            "channels": ";".join(image_info["channels_loaded"]),
            "features": all_features,
            # Diagnostic columns are merged later from the joined dataframe.
            "_row_meta": row,
        })

    return feature_records, profile_records, puncta_records, image_info


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_extract(
    *,
    masks_dir: Path,
    metadata_dir: Path,
    cell_table_path: Path,
    crop_manifest_path: Path | None,
    unet_classifications_path: Path | None,
    output_dir: Path,
    pixel_microns: float,
    channel_names: tuple[str, ...],
    phase_channel: str,
    feature_columns: tuple[str, ...] = (),
    include_unet_filter: bool = True,
    probability_threshold: float = 0.5,
    keep_class: str = "cell",
    require_momia_included: bool = True,
    padmap_path: Path | None = None,
    padmap_id_pattern: str | None = None,
    padmap_id_source: str = "source_path",
    padmap_join: str = "left",
    workers: int = 1,
    prefix: str = "sc_morph",
    segmentation_run_hash: str | None = None,
    fluorescence_config: dict | None = None,
    known_channels: tuple[str, ...] = KNOWN_CHANNELS,
) -> dict[str, Path]:
    """Run the deciphaer feature-extraction stage and emit final artifacts.

    Channels: pass an explicit tuple to lock the schema, or pass an empty
    tuple / ``("auto",)`` to auto-detect channels from sibling files of the
    first available Phase TIFF in the cell table (always includes
    ``phase_channel`` first).

    Feature columns: pass an explicit tuple to project a custom schema, or
    pass an empty tuple to emit every column produced for the resolved
    channels (the morphology block + intensity/profile/fluor-common for each
    channel + plugin extras for matching channels).

    See module docstring for the full output description.  Returns a dict
    of artifact name -> Path.
    """
    masks_dir = Path(masks_dir)
    metadata_dir = Path(metadata_dir)
    cell_table_path = Path(cell_table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from deciphaer_processing.hashing.cell_hash import parse_filename_metadata
    from deciphaer_processing.hashing.lookup import CellLookupTable
    from fluorescence.sidecar import write_puncta_parquet

    # ---- 1. Load + join cell table ---------------------------------------
    cells = pd.read_csv(cell_table_path)
    n_input = len(cells)
    _log.info("extract: loaded %d cells from %s", n_input, cell_table_path)

    if crop_manifest_path is not None and Path(crop_manifest_path).exists():
        crops = pd.read_csv(crop_manifest_path)
        cells = join_crops(cells, crops)
        _log.info("extract: joined %d crops -> %d rows", len(crops), len(cells))
    else:
        cells = _ensure_cell_ids(cells, segmentation_run_hash=segmentation_run_hash)

    # Make sure every row has cell_hash + Cell_ID.
    cells = _ensure_cell_provenance(cells)

    # ---- 2. Filtering ----------------------------------------------------
    if require_momia_included and "included" in cells.columns:
        included = cells["included"].astype(str).str.lower().isin({"true", "1", "yes"})
        before = len(cells)
        cells = cells.loc[included].copy()
        _log.info("extract: included filter kept %d/%d", len(cells), before)

    if unet_classifications_path is not None and Path(unet_classifications_path).exists():
        unet = pd.read_csv(unet_classifications_path)
        before = len(cells)
        cells = cells.merge(unet, on="Cell_ID", how="left", suffixes=("", "_unet"))
        if include_unet_filter:
            cells = filter_unet(
                cells,
                probability_threshold=probability_threshold,
                keep_class=keep_class,
            )
        _log.info("extract: U-Net merged (%d preds), kept %d/%d", len(unet), len(cells), before)

    if cells.empty:
        _log.warning("extract: no cells survived filtering — emitting empty outputs")

    # ---- 3. Per-image feature extraction --------------------------------
    cells["image_id"] = cells["image_id"].astype(str)
    cells["label"] = cells["label"].astype(int)

    # Resolve channels: explicit list wins; empty / ("auto",) triggers a scan
    # of the parent dir of the first available source path. The phase channel
    # is always pinned at index 0.
    channel_names = _resolve_channel_names(
        channel_names=channel_names,
        phase_channel=phase_channel,
        cells=cells,
        known_channels=tuple(known_channels),
    )
    _log.info("extract: channels resolved -> %s (phase=%s)", list(channel_names), phase_channel)

    # If no explicit feature_columns, derive them from the resolved channels.
    if not feature_columns:
        feature_columns = feature_columns_for_channels(
            channel_names, phase_channel=phase_channel,
        )
        _log.info("extract: feature_columns auto-derived (%d columns)", len(feature_columns))

    groups = list(cells.groupby("image_id", sort=False))
    tasks = [
        (
            str(image_id),
            group.to_dict("records"),
            str(masks_dir),
            float(pixel_microns),
            tuple(channel_names),
            str(phase_channel),
            dict(fluorescence_config or {}),
            tuple(known_channels),
        )
        for image_id, group in groups
    ]

    feature_records: list[dict[str, Any]] = []
    profile_records: list[dict[str, Any]] = []
    puncta_records: list[dict[str, Any]] = []
    image_infos: list[dict[str, Any]] = []

    n_total = len(tasks)
    log_every = max(1, n_total // 20)  # ~20 progress lines over the whole run
    extract_started = _time.monotonic()
    _log.info(
        "extract: feature extraction starting — %d images, %d cells, %d worker(s)",
        n_total, sum(len(t[1]) for t in tasks), max(1, workers),
    )

    def _log_progress(done: int) -> None:
        if done == 0:
            return
        if done == n_total or done % log_every == 0:
            elapsed = _time.monotonic() - extract_started
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / rate if rate > 0 else 0
            cells_so_far = len(feature_records)
            _log.info(
                "extract: %d/%d images done (%d cells extracted; %.1f img/s; ETA %.0fs)",
                done, n_total, cells_so_far, rate, eta,
            )

    if workers <= 1:
        for i, task in enumerate(tasks, start=1):
            try:
                f, p, q, info = _extract_image_worker(task)
            except Exception as exc:
                _log.warning("extract: image %s failed: %s", task[0], exc)
                _log_progress(i)
                continue
            feature_records.extend(f)
            profile_records.extend(p)
            puncta_records.extend(q)
            image_infos.append(info)
            _log_progress(i)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_extract_image_worker, task): task[0] for task in tasks}
            done = 0
            for future in as_completed(futures):
                done += 1
                image_id = futures[future]
                try:
                    f, p, q, info = future.result()
                except Exception as exc:
                    _log.warning("extract: image %s failed: %s", image_id, exc)
                    _log_progress(done)
                    continue
                feature_records.extend(f)
                profile_records.extend(p)
                puncta_records.extend(q)
                image_infos.append(info)
                _log_progress(done)

    _log.info("extract: feature extraction complete in %.1fs — %d cells, %d profile records, %d puncta records",
              _time.monotonic() - extract_started,
              len(feature_records), len(profile_records), len(puncta_records))

    # ---- 4. Build a dataframe of features + diagnostics -----------------
    _t = _time.monotonic()
    feat_df = _records_to_dataframe(feature_records)
    _log.info("extract: built feat_df in %.1fs (%d rows × %d cols)",
              _time.monotonic() - _t, len(feat_df), len(feat_df.columns))

    # Carry forward diagnostic / classification columns from `cells`.
    diag_cols = [
        "Cell_ID", "included", "touching_edge", "is_central", "failure_reasons",
        "predicted_class", "confidence", "p_background", "p_cell", "p_noncell",
        "predicted_cell",
    ]
    diag_subset = cells[[c for c in diag_cols if c in cells.columns]].copy()
    if "Cell_ID" in diag_subset.columns:
        feat_df = feat_df.merge(diag_subset, on="Cell_ID", how="left")
    else:
        for c in diag_cols:
            if c not in feat_df.columns:
                feat_df[c] = pd.NA

    feat_df["filter_passed"] = _compute_filter_passed(feat_df, probability_threshold, keep_class)

    # ---- 5. cell_lookup.parquet (pre-padmap, sc_<hash> ids) -------------
    # Build rows directly: deciphaer's CellLookupTable.add_cells_batch has a
    # schema bug (creates Int64 columns that fail to concat into the Int32
    # empty frame).  Going through the empty-schema constructor + writing the
    # parquet ourselves sidesteps it.
    _t = _time.monotonic()
    lookup = _build_cell_lookup(feat_df)
    _log.info("extract: built cell_lookup in %.1fs", _time.monotonic() - _t)

    # ---- 6. Padmap rename (drug + sc_<DRUG>_<seq>) ----------------------
    padmap_match_report: dict[str, Any] | None = None
    if padmap_path is not None and Path(padmap_path).exists():
        from deciphaer_processing.padmap.registry import (
            apply_padmap,
            derive_padmap_id,
            load_padmap,
        )
        import polars as pl

        padmap_df = load_padmap(padmap_path)
        meta_pl = pl.from_pandas(feat_df[[
            c for c in ("Cell_ID", "cell_hash", "source_path", "image_id")
            if c in feat_df.columns
        ]])
        try:
            meta_pl = derive_padmap_id(
                meta_pl,
                source=padmap_id_source,  # type: ignore[arg-type]
                regex=padmap_id_pattern,
            )
            joined, report = apply_padmap(
                meta_pl,
                padmap_df,
                join=padmap_join,  # type: ignore[arg-type]
            )
            padmap_match_report = report.to_dict()
            joined_pd = joined.to_pandas()
            # joined may not have Cell_ID if user passed a non-Cell_ID source col,
            # but our subset always includes it.
            feat_df = feat_df.drop(columns=[c for c in ("drug", "GroupLabels_Strict") if c in feat_df.columns])
            feat_df = feat_df.merge(
                joined_pd[[c for c in ("Cell_ID", "drug", "GroupLabels_Strict") if c in joined_pd.columns]],
                on="Cell_ID",
                how="left",
            )
        except Exception as exc:
            _log.warning("extract: padmap join failed (%s) — leaving drug=UNK", exc)
            feat_df["drug"] = feat_df.get("drug", "UNK")
            feat_df["GroupLabels_Strict"] = feat_df.get("GroupLabels_Strict", "UNK")
    else:
        feat_df["drug"] = feat_df.get("drug", "UNK")
        feat_df["GroupLabels_Strict"] = feat_df.get("GroupLabels_Strict", "UNK")

    feat_df["drug"] = feat_df.get("drug", "UNK")
    feat_df["drug"] = feat_df["drug"].fillna("UNK")
    if "GroupLabels_Strict" not in feat_df.columns:
        feat_df["GroupLabels_Strict"] = feat_df["drug"]
    else:
        feat_df["GroupLabels_Strict"] = feat_df["GroupLabels_Strict"].fillna(feat_df["drug"])

    # Rename Cell_IDs deterministically: sc_<DRUG>_<seq>.  Cells without a
    # matched drug stay as sc_<hash> so they remain traceable.
    feat_df = feat_df.sort_values(["source_filename", "label"], kind="stable").reset_index(drop=True)
    new_cell_ids = _assign_drug_cell_ids(feat_df)
    id_map = dict(zip(feat_df["Cell_ID"].tolist(), new_cell_ids))
    feat_df["Cell_ID"] = new_cell_ids

    # Rewrite cell_lookup with the new ids (cell_hash never changes).
    cell_lookup_path = output_dir / "cell_lookup.parquet"
    if lookup is not None and lookup.height > 0:
        import polars as pl

        new_ids_for_lookup = [
            id_map.get(f"sc_{h}", f"sc_{h}") for h in lookup["cell_hash"].to_list()
        ]
        lookup = lookup.with_columns(pl.Series("cell_id", new_ids_for_lookup))
        cell_lookup_path.parent.mkdir(parents=True, exist_ok=True)
        lookup.write_parquet(cell_lookup_path)
    else:
        _empty_cell_lookup(cell_lookup_path)

    # Update profile + puncta records with the new IDs.
    for rec in profile_records:
        rec["cell_id"] = id_map.get(rec["cell_id"], rec["cell_id"])
    for rec in puncta_records:
        rec["cell_id"] = id_map.get(rec["cell_id"], rec["cell_id"])

    # ---- 7. Build the four output tables --------------------------------
    _t = _time.monotonic()
    data_path = output_dir / f"{prefix}_data.csv"
    _write_data_csv(feat_df, feature_columns, data_path)
    _log.info("extract: wrote %s in %.1fs", data_path.name, _time.monotonic() - _t)

    _t = _time.monotonic()
    meta_path = output_dir / f"{prefix}_meta.csv"
    feat_df[["Cell_ID", "drug", "GroupLabels_Strict"]].to_csv(meta_path, index=False)
    _log.info("extract: wrote %s in %.1fs", meta_path.name, _time.monotonic() - _t)

    _t = _time.monotonic()
    mapping_path = output_dir / f"{prefix}_mapping.csv"
    _write_mapping_csv(feat_df, mapping_path, segmentation_run_hash=segmentation_run_hash)
    _log.info("extract: wrote %s in %.1fs", mapping_path.name, _time.monotonic() - _t)

    _t = _time.monotonic()
    diagnostics_path = output_dir / f"{prefix}_diagnostics.csv"
    _write_diagnostics_csv(feat_df, diagnostics_path)
    _log.info("extract: wrote %s in %.1fs", diagnostics_path.name, _time.monotonic() - _t)

    # ---- 8. Sidecar parquets --------------------------------------------
    # Profiles are always emitted (even when empty) so downstream consumers
    # can recreate per-cell fluorescence curves without re-running extract.
    # Puncta only emit when a plugin produced events.
    profiles_path = output_dir / f"{prefix}_profiles.parquet"
    _write_profiles_parquet_always(profile_records, profiles_path)
    puncta_path: Path | None = None
    if puncta_records:
        puncta_path = output_dir / f"{prefix}_puncta.parquet"
        write_puncta_parquet(puncta_records, puncta_path)

    # ---- 9. Summary ------------------------------------------------------
    summary_path = output_dir / "extract_summary.json"
    feature_count = len(feature_columns) if feature_columns else _count_emitted_features(feat_df)
    channels_seen = {info["image_id"]: info.get("channels_loaded", []) for info in image_infos}
    summary = {
        "n_cells_in": int(n_input),
        "n_cells_passed": int(len(feat_df)),
        "n_cells_after_padmap": int(len(feat_df)),
        "channels_seen": channels_seen,
        "feature_count": int(feature_count),
        "segmentation_run_hash": segmentation_run_hash or "",
        "padmap_match_report": padmap_match_report,
        "outputs": {
            "data": str(data_path),
            "meta": str(meta_path),
            "mapping": str(mapping_path),
            "diagnostics": str(diagnostics_path),
            "cell_lookup": str(cell_lookup_path),
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    artifacts: dict[str, Path] = {
        "data": data_path,
        "meta": meta_path,
        "mapping": mapping_path,
        "diagnostics": diagnostics_path,
        "cell_lookup": cell_lookup_path,
        "summary": summary_path,
        "profiles": profiles_path,
    }
    if puncta_path is not None:
        artifacts["puncta"] = puncta_path
    _log.info("extract: wrote %d artifacts to %s", len(artifacts), output_dir)
    return artifacts


# ---------------------------------------------------------------------------
# Helpers used by run_extract
# ---------------------------------------------------------------------------
_AUTO_CHANNEL_SENTINELS = frozenset({"auto", "AUTO", "Auto"})


def _resolve_channel_names(
    *,
    channel_names: tuple[str, ...],
    phase_channel: str,
    cells: pd.DataFrame,
    known_channels: tuple[str, ...] = KNOWN_CHANNELS,
) -> tuple[str, ...]:
    """Pick the final channel set used for this run.

    Explicit channels are passed through (with ``phase_channel`` pinned at
    index 0). An empty list, or a sentinel like ``("auto",)``, triggers
    auto-detection by scanning sibling TIFFs of the first available source
    path in ``cells``.
    """
    from deciphaer_image_segmentation.channels import discover_channels_in_dir

    explicit = [str(c) for c in channel_names if str(c) not in _AUTO_CHANNEL_SENTINELS]
    if explicit:
        if phase_channel not in explicit:
            explicit.insert(0, phase_channel)
        return tuple(dict.fromkeys(explicit))

    # Auto-detect: walk to the first existing source dir and probe it.
    if "source_path" not in cells.columns or cells.empty:
        return (phase_channel,)
    for value in cells["source_path"].dropna().astype(str).unique():
        try:
            parent = Path(value).resolve().parent
        except (OSError, ValueError):
            continue
        if parent.is_dir():
            return discover_channels_in_dir(
                parent,
                phase_channel=phase_channel,
                pattern=f"*{phase_channel}*.tif",
                recursive=False,
                known_channels=known_channels,
            )
    return (phase_channel,)


def _write_profiles_parquet_always(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write the midline-profiles sidecar, including when empty.

    Always emitting a parquet (even header-only) keeps downstream consumers
    simple: they can ``pd.read_parquet(path)`` unconditionally and not have
    to special-case ``FileNotFoundError`` for runs where no fluor channels
    were detected.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        pd.DataFrame(records).to_parquet(path, index=False)
        _log.info("extract: wrote %d profile records to %s", len(records), path)
        return
    empty = pd.DataFrame(
        {
            "cell_id": pd.Series(dtype="object"),
            "channel": pd.Series(dtype="object"),
            "profile_kind": pd.Series(dtype="object"),
            "n_points": pd.Series(dtype="int64"),
            "profile": pd.Series(dtype="object"),
        }
    )
    empty.to_parquet(path, index=False)
    _log.info("extract: wrote empty profiles parquet to %s", path)


def _ensure_cell_ids(cells: pd.DataFrame, *, segmentation_run_hash: str | None) -> pd.DataFrame:
    out = cells.copy()
    if "Cell_ID" in out.columns and out["Cell_ID"].notna().all():
        return out
    new_ids: list[str] = []
    new_hashes: list[str] = []
    for _, row in out.iterrows():
        bbox = _bbox_from_row(row)
        digest = _cell_hash_digest(Path(str(row.get("source_path", ""))), int(row.get("label", 0) or 0), bbox)
        new_ids.append(f"sc_{digest}")
        new_hashes.append(digest)
    out["Cell_ID"] = new_ids
    if "cell_hash" not in out.columns:
        out["cell_hash"] = new_hashes
    return out


def _ensure_cell_provenance(cells: pd.DataFrame) -> pd.DataFrame:
    out = cells.copy()
    if "Cell_ID" not in out.columns:
        out["Cell_ID"] = [
            f"sc_{_cell_hash_digest(Path(str(row.get('source_path', ''))), int(row.get('label', 0) or 0), _bbox_from_row(row))}"
            for _, row in out.iterrows()
        ]
    if "cell_hash" not in out.columns:
        out["cell_hash"] = out["Cell_ID"].astype(str).str.replace(r"^sc_", "", regex=True)
    else:
        # Strip any "sha256:" prefix the v2 manifest sometimes uses.
        out["cell_hash"] = out["cell_hash"].astype(str).str.replace(r"^sha256:", "", regex=True)
    if "source_filename" not in out.columns:
        out["source_filename"] = out["source_path"].astype(str).map(lambda p: Path(p).name)
    return out


def _compute_filter_passed(
    df: pd.DataFrame,
    probability_threshold: float,
    keep_class: str,
) -> pd.Series:
    passed = pd.Series(True, index=df.index)
    if "included" in df.columns:
        inc = df["included"].astype(str).str.lower().isin({"true", "1", "yes"})
        passed &= inc.fillna(False)
    if "p_cell" in df.columns:
        passed &= pd.to_numeric(df["p_cell"], errors="coerce").fillna(0) >= probability_threshold
    if keep_class == "cell" and "predicted_class" in df.columns:
        passed &= df["predicted_class"].astype(str).eq("cell")
    return passed


def _records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten the per-image records into a tidy per-cell dataframe."""
    if not records:
        cols = [
            "Cell_ID", "cell_hash", "image_id", "label", "source_path",
            "source_filename", "mask_path", "bbox_min_row", "bbox_min_col",
            "bbox_max_row", "bbox_max_col", "crop_path", "channels",
        ]
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for rec in records:
        row = {
            "Cell_ID": rec["Cell_ID"],
            "cell_hash": rec["cell_hash"],
            "image_id": rec["image_id"],
            "label": rec["label"],
            "source_path": rec["source_path"],
            "source_filename": rec["source_filename"],
            "mask_path": rec["mask_path"],
            "bbox_min_row": rec["bbox_min_row"],
            "bbox_min_col": rec["bbox_min_col"],
            "bbox_max_row": rec["bbox_max_row"],
            "bbox_max_col": rec["bbox_max_col"],
            "crop_path": rec.get("crop_path", ""),
            "channels": rec.get("channels", ""),
        }
        row.update(rec.get("features", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def _assign_drug_cell_ids(df: pd.DataFrame) -> list[str]:
    """Assign sc_<DRUG>_<seq> ids deterministically per drug group.

    Cells whose drug is empty/UNK fall back to ``sc_<hash>`` so they remain
    traceable.  Sort order (source_filename, label) is set by caller.
    """
    new_ids: list[str] = []
    counters: dict[str, int] = {}
    for _, row in df.iterrows():
        drug = row.get("drug")
        cell_hash = row.get("cell_hash") or ""
        if not isinstance(drug, str) or drug in ("", "UNK") or pd.isna(drug):
            new_ids.append(f"sc_{cell_hash}")
            continue
        counters[drug] = counters.get(drug, 0) + 1
        new_ids.append(f"sc_{drug}_{counters[drug]}")
    return new_ids


def _count_emitted_features(df: pd.DataFrame) -> int:
    blocked = {
        "Cell_ID", "cell_hash", "image_id", "label", "source_path",
        "source_filename", "mask_path", "bbox_min_row", "bbox_min_col",
        "bbox_max_row", "bbox_max_col", "crop_path", "channels",
        "included", "touching_edge", "is_central", "failure_reasons",
        "predicted_class", "confidence", "p_background", "p_cell", "p_noncell",
        "predicted_cell", "filter_passed", "drug", "GroupLabels_Strict",
    }
    return int(sum(1 for c in df.columns if c not in blocked))


def _write_data_csv(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    path: Path,
) -> None:
    """Write the wide ``feature, <Cell_ID>...`` matrix used by deciphaer.

    Vectorized: builds the (features × cells) matrix via ``df.reindex`` and
    transposes once, instead of nested per-cell-per-feature ``iloc.get``
    calls (which is O(C×F) Python-level lookups — for 40k cells × 176
    features that's 7M calls and dominates wall-clock).
    """
    if feature_columns:
        feature_list = list(feature_columns)
    else:
        blocked = {
            "Cell_ID", "cell_hash", "image_id", "label", "source_path",
            "source_filename", "mask_path", "bbox_min_row", "bbox_min_col",
            "bbox_max_row", "bbox_max_col", "crop_path", "channels",
            "included", "touching_edge", "is_central", "failure_reasons",
            "predicted_class", "confidence", "p_background", "p_cell", "p_noncell",
            "predicted_cell", "filter_passed", "drug", "GroupLabels_Strict",
        }
        feature_list = [c for c in df.columns if c not in blocked]

    cell_ids = df["Cell_ID"].astype(str).tolist()

    # Project to the requested feature columns; missing ones become NaN.
    feat_only = df.reindex(columns=feature_list)
    # Transpose so rows = features, columns = cells. Build the final frame
    # with a 'feature' column up front to match the reference schema.
    matrix = feat_only.T
    matrix.columns = cell_ids
    matrix.insert(0, "feature", feature_list)
    matrix.to_csv(path, index=False)


def _write_mapping_csv(df: pd.DataFrame, path: Path, *, segmentation_run_hash: str | None) -> None:
    """Build the full traceability mapping table.

    Avoids ``df.iterrows()`` (which boxes each row into a Series — slow on
    40k+ cells); plate/scene/position are parsed once per unique filename
    and broadcast back to all rows from that file (typically ~30 cells/file
    so this cuts ``parse_filename_metadata`` calls by ~30x).
    """
    from deciphaer_processing.hashing.cell_hash import parse_filename_metadata

    if df.empty:
        pd.DataFrame(
            columns=[
                "Cell_ID", "cell_hash", "source_path", "source_filename",
                "mask_label", "image_id", "plate", "scene", "position",
                "drug", "GroupLabels_Strict",
                "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col",
                "crop_path", "channels", "segmentation_run_hash",
            ],
        ).to_csv(path, index=False)
        return

    src = df.copy()
    if "source_filename" not in src.columns:
        src["source_filename"] = ""

    # Parse filename metadata once per unique filename, then merge.
    unique_files = src["source_filename"].astype(str).unique()
    fmeta = pd.DataFrame(
        [
            {"source_filename": f, **parse_filename_metadata(f)}
            for f in unique_files
        ]
    )
    src = src.merge(fmeta, on="source_filename", how="left")

    # Ensure all output columns exist; default missing ones to a sensible value.
    n = len(src)
    defaults: dict[str, Any] = {
        "Cell_ID": "", "cell_hash": "", "source_path": "",
        "label": "", "image_id": "",
        "plate": "UNK", "scene": "UNK", "position": "UNK",
        "drug": "UNK", "GroupLabels_Strict": "UNK",
        "bbox_min_row": "", "bbox_min_col": "", "bbox_max_row": "", "bbox_max_col": "",
        "crop_path": "", "channels": "",
    }
    for col, default in defaults.items():
        if col not in src.columns:
            src[col] = [default] * n

    out = pd.DataFrame({
        "Cell_ID": src["Cell_ID"],
        "cell_hash": src["cell_hash"],
        "source_path": src["source_path"],
        "source_filename": src["source_filename"],
        "mask_label": src["label"],
        "image_id": src["image_id"],
        "plate": src["plate"].fillna("UNK"),
        "scene": src["scene"].fillna("UNK"),
        "position": src["position"].fillna("UNK"),
        "drug": src["drug"].fillna("UNK"),
        "GroupLabels_Strict": src["GroupLabels_Strict"].fillna("UNK"),
        "bbox_min_row": src["bbox_min_row"],
        "bbox_min_col": src["bbox_min_col"],
        "bbox_max_row": src["bbox_max_row"],
        "bbox_max_col": src["bbox_max_col"],
        "crop_path": src["crop_path"],
        "channels": src["channels"],
        "segmentation_run_hash": segmentation_run_hash or "",
    })
    out.to_csv(path, index=False)


def _write_diagnostics_csv(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "Cell_ID", "filter_passed", "included", "touching_edge", "is_central",
        "failure_reasons", "predicted_class", "confidence",
        "p_background", "p_cell", "p_noncell", "predicted_cell",
    ]
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()
    # Always include Cell_ID first.
    if "Cell_ID" in out.columns and present[0] != "Cell_ID":
        present.remove("Cell_ID")
        out = pd.concat([df[["Cell_ID"]], out.drop(columns=["Cell_ID"])], axis=1)
    # Pad any missing columns with NaN so downstream consumers see a stable schema.
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols]
    out.to_csv(path, index=False)


def _build_cell_lookup(feat_df: pd.DataFrame):
    """Build a polars cell_lookup DataFrame matching the reference schema.

    Mirrors deciphaer's ``CellLookupTable`` schema (Int32 ints, Datetime).
    Uses ``parse_filename_metadata`` for plate/scene/position parsing.
    """
    import polars as pl

    from deciphaer_processing.hashing.cell_hash import parse_filename_metadata

    schema = {
        "cell_hash": pl.Utf8,
        "cell_id": pl.Utf8,
        "source_path": pl.Utf8,
        "source_filename": pl.Utf8,
        "mask_label": pl.Int32,
        "bbox_min_row": pl.Int32,
        "bbox_min_col": pl.Int32,
        "bbox_max_row": pl.Int32,
        "bbox_max_col": pl.Int32,
        "plate": pl.Utf8,
        "scene": pl.Utf8,
        "position": pl.Utf8,
        "timestamp": pl.Datetime,
    }
    if feat_df is None or feat_df.empty:
        return pl.DataFrame(schema=schema)

    # Vectorized: avoid per-row iterrows on 40k+ cells. Parse plate/scene/
    # position once per unique filename and merge.
    src = feat_df.sort_values(["source_filename", "label"], kind="stable").copy()
    if "source_filename" not in src.columns:
        src["source_filename"] = src.get("source_path", "").astype(str).map(
            lambda p: Path(p).name
        )
    src["source_filename"] = src["source_filename"].fillna("").astype(str)
    unique_files = src["source_filename"].unique()
    fmeta = pd.DataFrame(
        [
            {"source_filename": f, **parse_filename_metadata(f)}
            for f in unique_files
        ]
    )
    src = src.merge(fmeta, on="source_filename", how="left")

    n = len(src)
    now = datetime.utcnow()
    out_pd = pd.DataFrame({
        "cell_hash": src.get("cell_hash", "").fillna("").astype(str),
        "cell_id": src.get("Cell_ID", "").fillna("").astype(str),
        "source_path": src.get("source_path", "").fillna("").astype(str),
        "source_filename": src["source_filename"],
        "mask_label": pd.to_numeric(src.get("label", 0), errors="coerce").fillna(0).astype("int32"),
        "bbox_min_row": pd.to_numeric(src.get("bbox_min_row", 0), errors="coerce").fillna(0).astype("int32"),
        "bbox_min_col": pd.to_numeric(src.get("bbox_min_col", 0), errors="coerce").fillna(0).astype("int32"),
        "bbox_max_row": pd.to_numeric(src.get("bbox_max_row", 0), errors="coerce").fillna(0).astype("int32"),
        "bbox_max_col": pd.to_numeric(src.get("bbox_max_col", 0), errors="coerce").fillna(0).astype("int32"),
        "plate": src.get("plate", "UNK").fillna("UNK").astype(str),
        "scene": src.get("scene", "UNK").fillna("UNK").astype(str),
        "position": src.get("position", "UNK").fillna("UNK").astype(str),
        "timestamp": [now] * n,
    })
    return pl.from_pandas(out_pd, schema_overrides={
        "mask_label": pl.Int32,
        "bbox_min_row": pl.Int32, "bbox_min_col": pl.Int32,
        "bbox_max_row": pl.Int32, "bbox_max_col": pl.Int32,
    })


def _empty_cell_lookup(path: Path) -> None:
    """Write an empty cell_lookup.parquet matching the deciphaer schema."""
    import polars as pl

    df = pl.DataFrame(
        schema={
            "cell_hash": pl.Utf8,
            "cell_id": pl.Utf8,
            "source_path": pl.Utf8,
            "source_filename": pl.Utf8,
            "mask_label": pl.Int32,
            "bbox_min_row": pl.Int32,
            "bbox_min_col": pl.Int32,
            "bbox_max_row": pl.Int32,
            "bbox_max_col": pl.Int32,
            "plate": pl.Utf8,
            "scene": pl.Utf8,
            "position": pl.Utf8,
            "timestamp": pl.Datetime,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
