from __future__ import annotations

import csv
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import tifffile

from deciphaer_image_segmentation.channels import (
    KNOWN_CHANNELS,
    discover_sibling_channels,
    parse_channel_token,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CropRecord:
    Cell_ID: str
    cell_hash: str
    segmentation_run_hash: str
    original_cell_id: str
    image_id: str
    source_image: str
    source_path: str
    mask_path: str
    label: int
    crop_path: str
    bbox_min_row: int
    bbox_min_col: int
    bbox_max_row: int
    bbox_max_col: int
    crop_size_y: int
    crop_size_x: int
    channels: str
    channel_metadata: str


def crop_cells_from_table(
    *,
    cell_table_path: Path,
    masks_dir: Path,
    crops_dir: Path,
    selected_channels: tuple[str, ...],
    phase_channel: str,
    crop_size: int,
    min_bbox_pad: int,
    edge_pad_mode: str,
    workers: int,
    include_available_channels: bool = True,
    segmentation_run_hash: str | None = None,
    shard: tuple[int, int] | None = None,
    known_channels: tuple[str, ...] = KNOWN_CHANNELS,
) -> Path:
    raw = pd.read_csv(cell_table_path)
    n_total = len(raw)
    cells = raw
    if "included" in cells.columns:
        included = cells["included"].astype(str).str.lower().isin({"true", "1", "yes"})
        cells = cells.loc[included].copy()
    crops_dir.mkdir(parents=True, exist_ok=True)
    n_images = cells["image_id"].nunique() if "image_id" in cells.columns else 0
    _log.info(
        "crop: %d/%d cells included across %d images — crop_size=%d, workers=%d, channels=%s",
        len(cells),
        n_total,
        n_images,
        crop_size,
        workers,
        list(selected_channels),
    )

    groups = list(cells.groupby("image_id", sort=False))
    if shard is not None:
        k, n = shard
        total_images = len(groups)
        groups = groups[k::n]
        _log.info("crop: shard %d/%d takes %d of %d images", k, n, len(groups), total_images)

    # Pre-scan only the parent dirs of THIS shard's images so we don't pay the
    # listdir cost for images other shards will handle.
    if groups and include_available_channels:
        sharded_cells = pd.concat([g for _, g in groups], ignore_index=True)
        parent_listings = _build_parent_listings(sharded_cells)
    else:
        parent_listings = {}

    tasks = [
        (
            str(image_id),
            group.to_dict("records"),
            str(masks_dir),
            str(crops_dir),
            selected_channels,
            phase_channel,
            crop_size,
            min_bbox_pad,
            edge_pad_mode,
            include_available_channels,
            segmentation_run_hash or "",
            parent_listings,
            tuple(known_channels),
        )
        for image_id, group in groups
    ]

    records: list[CropRecord] = []
    completed_images = 0
    log_step = max(1, len(tasks) // 10)
    if workers <= 1:
        for task in tasks:
            records.extend(_crop_image_worker(task))
            completed_images += 1
            if completed_images % log_step == 0:
                _log.info("crop: processed %d/%d images", completed_images, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_crop_image_worker, task) for task in tasks]
            for future in as_completed(futures):
                records.extend(future.result())
                completed_images += 1
                if completed_images % log_step == 0:
                    _log.info("crop: processed %d/%d images", completed_images, len(tasks))

    records.sort(key=lambda item: (item.image_id, item.label, item.Cell_ID))
    if shard is not None:
        k, n = shard
        manifest_name = f"crops_manifest_shard_{k}_of_{n}.csv"
    else:
        manifest_name = "crops_manifest.csv"
    manifest = crops_dir / manifest_name
    write_crop_manifest(manifest, records)
    _log.info("crop: wrote %d crops to %s, manifest=%s", len(records), crops_dir, manifest)
    return manifest


def merge_crop_manifests(crops_dir: Path, *, n_shards: int) -> Path:
    frames = []
    for k in range(n_shards):
        path = crops_dir / f"crops_manifest_shard_{k}_of_{n_shards}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No crop manifest shards found in {crops_dir}")
    merged = pd.concat(frames, ignore_index=True)
    out = crops_dir / "crops_manifest.csv"
    merged.to_csv(out, index=False)
    return out


def _build_parent_listings(cells: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """Pre-scan each unique source-image parent dir once.

    Returns parent_dir → list of (stem, resolved_path) for every TIFF in that dir.
    Workers can scan this list instead of doing their own iterdir per image,
    which matters when a single dir holds thousands of files.
    """
    listings: dict[str, list[tuple[str, str]]] = {}
    if "source_path" not in cells.columns:
        return listings
    parents: set[Path] = set()
    for value in cells["source_path"].dropna().astype(str).unique():
        try:
            parents.add(Path(value).resolve().parent)
        except (OSError, ValueError):
            continue
    for parent in parents:
        if not parent.is_dir():
            continue
        entries: list[tuple[str, str]] = []
        for candidate in parent.iterdir():
            if candidate.is_file() and candidate.suffix.lower() in {".tif", ".tiff"}:
                entries.append((candidate.stem, str(candidate.resolve())))
        listings[str(parent)] = entries
    return listings


def write_filtered_composites(
    *,
    cell_table_path: Path,
    masks_dir: Path,
    composites_dir: Path,
    selected_channels: tuple[str, ...],
    phase_channel: str,
    include_available_channels: bool = True,
    segmentation_run_hash: str | None = None,
    known_channels: tuple[str, ...] = KNOWN_CHANNELS,
) -> Path:
    df = pd.read_csv(cell_table_path)
    if "included" in df.columns:
        included = df["included"].astype(str).str.lower().isin({"true", "1", "yes"})
        df = df.loc[included].copy()
    composites_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for image_id, group in df.groupby("image_id"):
        source = Path(str(group["source_path"].iloc[0])).resolve()
        mask_path = masks_dir / f"{image_id}_mask.tif"
        if not mask_path.exists():
            continue
        mask = _read_2d(mask_path)
        labels = set(int(v) for v in group["label"].astype(int))
        filtered_mask = np.where(np.isin(mask, list(labels)), mask, 0).astype(mask.dtype)
        channel_arrays = []
        labels_out = []
        channel_metadata = []
        for channel, path in _channel_paths_for_source(
            source=source,
            selected_channels=selected_channels,
            phase_channel=phase_channel,
            include_available_channels=include_available_channels,
            known_channels=known_channels,
        ):
            channel_arrays.append(_read_2d(path))
            labels_out.append(channel)
            channel_metadata.append({"label": channel, "kind": "image", "path": str(path)})
        channel_arrays.append(filtered_mask)
        labels_out.append("Mask")
        channel_metadata.append({"label": "Mask", "kind": "mask", "path": str(mask_path)})
        stack = _stack_uint16(channel_arrays)
        out_path = composites_dir / f"{_base_from_phase(source, phase_channel, known_channels=known_channels)}.tif"
        tifffile.imwrite(
            str(out_path),
            stack,
            ome=True,
            metadata=_ome_metadata(
                labels=labels_out,
                info={
                    "image_id": str(image_id),
                    "source_image": _base_from_phase(source, phase_channel, known_channels=known_channels),
                    "source_path": str(source),
                    "mask_path": str(mask_path),
                    "segmentation_run_hash": segmentation_run_hash or "",
                    "channels": channel_metadata,
                },
                name=_base_from_phase(source, phase_channel, known_channels=known_channels),
                description="deciphaer filtered image composite",
            ),
        )
        rows.append(
            {
                "image_id": str(image_id),
                "source_image": _base_from_phase(source, phase_channel, known_channels=known_channels),
                "composite_path": str(out_path),
                "n_labels": str(len(labels)),
                "channels": ";".join(labels_out),
                "segmentation_run_hash": segmentation_run_hash or "",
            }
        )
    manifest = composites_dir / "composites_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "source_image",
                "composite_path",
                "n_labels",
                "channels",
                "segmentation_run_hash",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def write_crop_manifest(path: Path, records: list[CropRecord]) -> None:
    fieldnames = list(CropRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def _crop_image_worker(args: tuple[Any, ...]) -> list[CropRecord]:
    """Process every cell of a single image with one mask + channel read each.

    Previously each cell paid the I/O cost of re-reading its image; for crowded
    images that was 100x+ redundant. Grouping per `image_id` cuts that cleanly.
    """
    (
        image_id,
        rows,
        masks_dir,
        crops_dir,
        selected_channels,
        phase_channel,
        crop_size,
        min_bbox_pad,
        edge_pad_mode,
        include_available_channels,
        segmentation_run_hash,
        parent_listings,
        known_channels,
    ) = args
    if not rows:
        return []

    source = Path(str(rows[0]["source_path"])).resolve()
    mask_path = Path(masks_dir) / f"{image_id}_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    mask = _read_2d(mask_path)

    parent_listing = parent_listings.get(str(source.parent), []) if include_available_channels else []
    channel_pairs = _channel_paths_for_source(
        source=source,
        selected_channels=selected_channels,
        phase_channel=phase_channel,
        include_available_channels=bool(include_available_channels),
        parent_listing=parent_listing,
        known_channels=known_channels,
    )
    channel_arrays = [(channel, _read_2d(path), path) for channel, path in channel_pairs]
    source_image = _base_from_phase(source, phase_channel, known_channels=known_channels)

    out: list[CropRecord] = []
    for row in rows:
        label = int(row["label"])
        bbox = _bbox_from_row(row)
        y0, y1, x0, x1 = _target_window(
            row=row,
            bbox=bbox,
            mask_shape=mask.shape,
            crop_size=int(crop_size),
            min_bbox_pad=int(min_bbox_pad),
        )

        stack_arrays: list[np.ndarray] = []
        written_channels: list[str] = []
        channel_metadata: list[dict[str, str]] = []
        for channel, image, channel_path in channel_arrays:
            stack_arrays.append(_extract_window(image, y0, y1, x0, x1, pad_mode=edge_pad_mode, fill_value=0))
            written_channels.append(channel)
            channel_metadata.append({"label": channel, "kind": "image", "path": str(channel_path)})

        mask_crop = _extract_window(mask, y0, y1, x0, x1, pad_mode="constant", fill_value=0)
        mask_crop = np.where(mask_crop == label, label, 0).astype(mask.dtype)
        stack_arrays.append(mask_crop)
        written_channels.append("Mask")
        channel_metadata.append({"label": "Mask", "kind": "mask", "path": str(mask_path)})
        stack = _stack_uint16(stack_arrays)

        cell_hash_digest = _cell_hash(
            source_path=source,
            label=label,
            bbox=bbox,
            segmentation_run_hash=str(segmentation_run_hash),
        )
        cell_hash = f"sha256:{cell_hash_digest}"
        cell_id = f"sc_{cell_hash_digest}"
        crop_path = Path(crops_dir) / f"{cell_id}.tif"
        tifffile.imwrite(
            str(crop_path),
            stack,
            ome=True,
            metadata=_ome_metadata(
                labels=written_channels,
                info={
                    "Cell_ID": cell_id,
                    "cell_hash": cell_hash,
                    "segmentation_run_hash": str(segmentation_run_hash),
                    "source_image": source_image,
                    "source_path": str(source),
                    "image_id": image_id,
                    "mask_path": str(mask_path),
                    "label": label,
                    "bbox": {
                        "min_row": bbox[0],
                        "min_col": bbox[1],
                        "max_row": bbox[2],
                        "max_col": bbox[3],
                    },
                    "channels": channel_metadata,
                },
                name=cell_id,
                description="deciphaer single-cell crop",
            ),
        )
        out.append(
            CropRecord(
                Cell_ID=cell_id,
                cell_hash=cell_hash,
                segmentation_run_hash=str(segmentation_run_hash),
                original_cell_id=str(row.get("cell_id", f"{image_id}_{label}")),
                image_id=image_id,
                source_image=source_image,
                source_path=str(source),
                mask_path=str(mask_path),
                label=label,
                crop_path=str(crop_path),
                bbox_min_row=bbox[0],
                bbox_min_col=bbox[1],
                bbox_max_row=bbox[2],
                bbox_max_col=bbox[3],
                crop_size_y=int(stack.shape[1]),
                crop_size_x=int(stack.shape[2]),
                channels=";".join(written_channels),
                channel_metadata=json.dumps(channel_metadata, sort_keys=True),
            )
        )
    return out


def _bbox_from_row(row: dict[str, Any]) -> tuple[int, int, int, int]:
    x1 = int(float(row.get("bbox_x1", row.get("bbox_min_col", 0))))
    y1 = int(float(row.get("bbox_y1", row.get("bbox_min_row", 0))))
    x2 = int(float(row.get("bbox_x2", row.get("bbox_max_col", x1 + 1))))
    y2 = int(float(row.get("bbox_y2", row.get("bbox_max_row", y1 + 1))))
    return y1, x1, y2, x2


def _target_window(
    *,
    row: dict[str, Any],
    bbox: tuple[int, int, int, int],
    mask_shape: tuple[int, int],
    crop_size: int,
    min_bbox_pad: int,
) -> tuple[int, int, int, int]:
    y0, x0, y1, x1 = bbox
    bbox_h = max(1, y1 - y0)
    bbox_w = max(1, x1 - x0)
    win_h = max(crop_size, bbox_h + 2 * min_bbox_pad)
    win_w = max(crop_size, bbox_w + 2 * min_bbox_pad)
    if "centroid_y" in row and "centroid_x" in row:
        cy = int(round(float(row["centroid_y"])))
        cx = int(round(float(row["centroid_x"])))
    else:
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
    top = cy - win_h // 2
    left = cx - win_w // 2
    return top, top + win_h, left, left + win_w


def _extract_window(
    image: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    *,
    pad_mode: str,
    fill_value: int,
) -> np.ndarray:
    h, w = image.shape
    ey0, ey1 = max(0, y0), min(h, y1)
    ex0, ex1 = max(0, x0), min(w, x1)
    crop = image[ey0:ey1, ex0:ex1]
    pad_top = ey0 - y0
    pad_bottom = y1 - ey1
    pad_left = ex0 - x0
    pad_right = x1 - ex1
    if not any((pad_top, pad_bottom, pad_left, pad_right)):
        return crop.copy()
    pads = ((pad_top, pad_bottom), (pad_left, pad_right))
    if pad_mode == "constant":
        return np.pad(crop, pads, mode="constant", constant_values=fill_value)
    return np.pad(crop, pads, mode="reflect")


def _cell_hash(
    *,
    source_path: Path,
    label: int,
    bbox: tuple[int, int, int, int],
    segmentation_run_hash: str,
) -> str:
    raw = (
        f"{segmentation_run_hash}|{source_path.resolve()}|{label}|"
        f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _ome_metadata(
    *,
    labels: list[str],
    info: dict[str, Any],
    name: str,
    description: str,
) -> dict[str, Any]:
    provenance_json = json.dumps(info, sort_keys=True)
    return {
        "axes": "CYX",
        "Name": name,
        "Description": description,
        "Channel": {"Name": labels},
        "MapAnnotation": {
            "Namespace": "deciphaer:image-segmentation",
            "Value": {
                "deciphaer.provenance_json": provenance_json,
                "deciphaer.channel_labels": ";".join(labels),
            },
        },
    }


def _channel_paths_for_source(
    *,
    source: Path,
    selected_channels: tuple[str, ...],
    phase_channel: str,
    include_available_channels: bool,
    parent_listing: list[tuple[str, str]] | None = None,
    known_channels: Iterable[str] = KNOWN_CHANNELS,
) -> list[tuple[str, Path]]:
    known = tuple(known_channels)
    channels = [channel for channel in selected_channels if channel != "Mask"]
    if phase_channel not in channels:
        channels.insert(0, phase_channel)
    discovered: dict[str, Path] = {}
    if include_available_channels:
        discovered = discover_sibling_channels(
            source,
            phase_channel=phase_channel,
            known_channels=known,
            listing=parent_listing,
        )
        for channel in sorted(discovered):
            if channel not in channels:
                channels.append(channel)
    return [
        (
            channel,
            discovered.get(channel) or _resolve_channel_path(
                source, channel=channel, phase_channel=phase_channel, known_channels=known,
            ),
        )
        for channel in dict.fromkeys(channels)
    ]


def _resolve_channel_path(
    source: Path,
    *,
    channel: str,
    phase_channel: str,
    known_channels: Iterable[str] = KNOWN_CHANNELS,
) -> Path:
    """Find ``channel``'s file next to ``source`` (the Phase TIFF).

    Tries the registry-based parser first (handles trailing tags like
    ``_ORG`` and spaced qualifiers like ``Phase PH3``), then falls back to
    the legacy strict-suffix / name-substitution lookups so unchanged old
    naming still works.
    """
    if channel == phase_channel:
        return source
    known = tuple(known_channels)
    siblings = discover_sibling_channels(
        source, phase_channel=phase_channel, known_channels=known,
    )
    if channel in siblings:
        return siblings[channel]
    # Legacy fallbacks for `<base>_<channel>.tif` strict naming.
    phase_suffix = f"_{phase_channel}"
    if source.stem.endswith(phase_suffix):
        base = source.stem[: -len(phase_suffix)]
        candidate = source.with_name(f"{base}_{channel}{source.suffix}")
        if candidate.exists():
            return candidate
    candidate = source.with_name(source.name.replace(phase_channel, channel))
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find channel {channel!r} next to {source}")


def _base_from_phase(
    source: Path,
    phase_channel: str,
    *,
    known_channels: Iterable[str] = KNOWN_CHANNELS,
) -> str:
    """Return the part of the filename stem preceding the channel token.

    Uses the registry-based parser so quirky exports (e.g. trailing ``_ORG``)
    still yield a clean base.
    """
    _, base = parse_channel_token(
        source.stem, known_channels=tuple(known_channels), phase_channel=phase_channel,
    )
    return base


def _read_2d(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF at {path}, got {arr.shape}")
    return arr


def _stack_uint16(arrays: list[np.ndarray]) -> np.ndarray:
    shape = arrays[0].shape
    stack = np.empty((len(arrays), *shape), dtype=np.uint16)
    for idx, arr in enumerate(arrays):
        if arr.shape != shape:
            raise ValueError(f"Channel shape mismatch: {arr.shape} vs {shape}")
        if arr.dtype == np.uint16:
            stack[idx] = arr
        elif np.issubdtype(arr.dtype, np.integer):
            stack[idx] = np.clip(arr, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        else:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0 or float(finite.max()) <= float(finite.min()):
                stack[idx] = np.zeros(shape, dtype=np.uint16)
            else:
                scaled = (arr - float(finite.min())) / (float(finite.max()) - float(finite.min()))
                stack[idx] = np.clip(scaled * 65535, 0, 65535).astype(np.uint16)
    return stack
