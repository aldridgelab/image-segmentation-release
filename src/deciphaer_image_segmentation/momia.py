from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from pandas.errors import EmptyDataError

from momia_seg.batch import DEFAULT_SETTINGS, deep_merge, run_batch, validate_settings

from deciphaer_image_segmentation.batching import read_batch_manifest
from deciphaer_image_segmentation.config import PipelineConfig
from deciphaer_image_segmentation.spatial import scale_momia_settings

_log = logging.getLogger(__name__)


def effective_momia_settings(config: PipelineConfig) -> dict[str, Any]:
    settings = deep_merge(DEFAULT_SETTINGS, config.momia.settings)
    # The pipeline input calibration is authoritative, including for validation.
    settings["image"]["pixel_microns"] = config.inputs.pixel_microns
    validate_settings(settings)
    settings = scale_momia_settings(settings, config.spatial_scale)
    validate_settings(settings)
    return settings


def run_momia_batch(config: PipelineConfig, batch_name: str) -> Path:

    batch_input = config.batch_dir / batch_name
    batch_output = config.momia_batches_dir / batch_name
    settings = effective_momia_settings(config)
    settings = deep_merge(
        settings,
        {
            "io": {
                "input_dir": str(batch_input),
                "output_dir": str(batch_output),
                "pattern": config.inputs.pattern,
                "recursive": False,
            },
            "image": {
                "channel_names": list(config.inputs.channels),
                "channel_axis": config.inputs.channel_axis,
                "ref_channel": config.inputs.phase_channel,
                "pixel_microns": config.inputs.pixel_microns,
            },
            "runtime": {
                "workers": config.momia.workers_per_batch,
                "force": config.momia.force,
                "skip_existing": config.momia.skip_existing,
                "fail_fast": config.momia.fail_fast,
                "dry_run": config.run.dry_run,
                "log_level": config.run.log_level,
            },
        },
    )
    validate_settings(settings)
    batch_output.mkdir(parents=True, exist_ok=True)
    settings_path = batch_output / "effective_momia_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    n_input_images = sum(1 for p in batch_input.glob("*") if p.is_file() or p.is_symlink())
    _log.info(
        "momia[%s]: input=%s (%d images), output=%s, workers=%d, skip_existing=%s",
        batch_name,
        batch_input,
        n_input_images,
        batch_output,
        config.momia.workers_per_batch,
        config.momia.skip_existing,
    )
    summary = run_batch(settings=settings, settings_path=settings_path)
    summary_path = batch_output / "v2_momia_summary.json"
    summary_path.write_text(
        json.dumps({**asdict(summary), "results": [asdict(r) for r in summary.results]}, indent=2),
        encoding="utf-8",
    )
    _log.info(
        "momia[%s]: processed=%d (succeeded=%d, skipped=%d, failed=%d) — wrote %s",
        batch_name,
        summary.processed,
        summary.succeeded,
        summary.skipped,
        summary.failed,
        summary_path,
    )
    return batch_output


def compile_momia_batches(
    config: PipelineConfig,
    *,
    link_mode: str = "symlink",
    log_dir: Path | None = None,
    skip_path: Path | None = None,
) -> Path:
    compiled = config.momia_compiled_dir
    masks_dir = compiled / "masks"
    metadata_dir = compiled / "metadata"
    masks_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    _log.info(
        "compile: scanning %d batches under %s (link_mode=%s)",
        len(config.batch_names),
        config.momia_batches_dir,
        link_mode,
    )

    # Watchdog: compile is wired in slurm with `afterany` on the MOMIA
    # batches so a single failed/OOM batch can't wedge the whole DAG.
    # Detect any failed batches, post one webhook listing them, and flag
    # their images into skip.txt before continuing on with whatever did
    # succeed. Cheap to skip when there's no slurm manifest (local runs).
    from deciphaer_image_segmentation.watchdog import flag_failed_batches

    flagged = flag_failed_batches(
        config,
        log_dir=log_dir,
        skip_path=skip_path or (config.output_dir / "skip.txt"),
        webhook_url=config.notifications.webhook_url,
        context="compile-watchdog",
    )
    if flagged:
        _log.warning(
            "compile: watchdog flagged %d image(s) from failed batches; "
            "continuing with surviving cells",
            len(flagged),
        )

    source_lookup = _source_lookup(config)
    summaries: list[dict[str, Any]] = []
    cell_tables: list[pd.DataFrame] = []
    n_with_summary = 0
    n_missing = 0
    n_corrupt = 0
    for batch_name in config.batch_names:
        batch_out = config.momia_batches_dir / batch_name
        summary_path = batch_out / "run_summary.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                _log.warning("compile: skipping malformed %s — %s", summary_path, exc)
                n_corrupt += 1
                payload = None
            if payload is not None:
                payload["batch"] = batch_name
                summaries.append(payload)
                n_with_summary += 1
        elif batch_out.exists():
            n_missing += 1
        _transfer_tree(batch_out / "masks", masks_dir, mode=link_mode)
        _transfer_tree(batch_out / "metadata", metadata_dir, mode=link_mode)
        df = _read_batch_cell_table(batch_out)
        if df is not None and not df.empty:
            df.insert(0, "batch", batch_name)
            if "source_path" in df.columns:
                df["source_path"] = df["source_path"].map(lambda value: source_lookup.get(str(value), str(Path(value).resolve())))
            cell_tables.append(df)

    if cell_tables:
        cells = pd.concat(cell_tables, ignore_index=True)
    else:
        cells = pd.DataFrame()
    cells_path = compiled / "cell_measurements.csv"
    cells.to_csv(cells_path, index=False)

    n_masks = sum(1 for _ in masks_dir.glob("*_mask.tif"))
    n_meta = sum(1 for _ in metadata_dir.glob("*_meta.json"))
    if n_missing:
        _log.warning("compile: %d batch(es) had no run_summary.json — partial / interrupted runs?", n_missing)
    if n_corrupt:
        _log.warning("compile: %d batch(es) had malformed run_summary.json (likely concurrent writes) — using metadata only", n_corrupt)
    _log.info(
        "compile: ingested %d/%d batches with summaries — %d cell rows, %d masks, %d metadata files",
        n_with_summary,
        len(config.batch_names),
        len(cells),
        n_masks,
        n_meta,
    )

    aggregate = {
        "compiled_at": pd.Timestamp.utcnow().isoformat(),
        "n_batches": len(config.batch_names),
        "n_cell_rows": int(len(cells)),
        "batch_summaries": summaries,
        "artifacts": {
            "cell_measurements": str(cells_path),
            "masks_dir": str(masks_dir),
            "metadata_dir": str(metadata_dir),
        },
    }
    (compiled / "compile_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    _log.info("compile: wrote %s", cells_path)
    return compiled


def _source_lookup(config: PipelineConfig) -> dict[str, str]:
    manifest = config.batch_dir / "batch_manifest.csv"
    if not manifest.exists():
        return {}
    out: dict[str, str] = {}
    for row in read_batch_manifest(manifest):
        out[row.link_path] = row.source_path
        out[str(Path(row.link_path).resolve())] = row.source_path
    return out


def _transfer_tree(src_dir: Path, dst_dir: Path, *, mode: str) -> None:
    if not src_dir.exists():
        return
    for src in sorted(src_dir.glob("*")):
        if not src.is_file():
            continue
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "hardlink":
            os.link(src, dst)
        else:
            dst.symlink_to(src.resolve())


def _read_batch_cell_table(batch_out: Path) -> pd.DataFrame | None:
    cells_path = batch_out / "cell_measurements.csv"
    if cells_path.exists():
        try:
            df = pd.read_csv(cells_path)
        except EmptyDataError:
            df = pd.DataFrame()
        if not df.empty:
            return df
    metadata_rows = _cells_from_metadata(batch_out / "metadata")
    return pd.DataFrame(metadata_rows) if metadata_rows else None


def _cells_from_metadata(metadata_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not metadata_dir.exists():
        return rows
    for metadata_path in sorted(metadata_dir.glob("*_meta.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cells = metadata.get("cells", [])
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if isinstance(cell, Mapping):
                rows.append(_flatten_cell_row(metadata=metadata, cell=cell))
    return rows


def _flatten_cell_row(*, metadata: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    image_id = str(metadata.get("image_id", metadata.get("id", "")))
    row: dict[str, Any] = {
        "image_id": image_id,
        "source_path": str(metadata.get("source_path", "")),
        "label": cell.get("label"),
        "cell_id": cell.get("cell_id", f"{image_id}_{cell.get('label', '')}"),
        "ucid": cell.get("ucid"),
        "included": cell.get("included"),
        "is_central": cell.get("is_central"),
        "touching_edge": cell.get("touching_edge"),
        "failure_reasons": "; ".join(str(item) for item in cell.get("failure_reasons", [])),
    }
    bbox = cell.get("bbox", [])
    if isinstance(bbox, list) and len(bbox) == 4:
        row.update({"bbox_x1": bbox[0], "bbox_y1": bbox[1], "bbox_x2": bbox[2], "bbox_y2": bbox[3]})
    centroid = cell.get("centroid", [])
    if isinstance(centroid, list) and len(centroid) == 2:
        row.update({"centroid_y": centroid[0], "centroid_x": centroid[1]})
    metrics = cell.get("metrics", {})
    if isinstance(metrics, Mapping):
        for key, value in metrics.items():
            if isinstance(value, Mapping) and key == "intensity":
                for channel, channel_value in value.items():
                    row[f"intensity_{channel}"] = channel_value
            elif not isinstance(value, (Mapping, list)):
                row[key] = value
    return row
