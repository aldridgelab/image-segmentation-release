"""Settings-sweep runner for comparing MOMIA2 segmentation configurations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from .batch import (
    deep_merge,
    load_json_file,
    resolve_effective_settings,
    run_batch,
    validate_settings,
)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "run"


def _set_if_present(overrides: dict[str, Any], section: str, key: str, value: Any) -> None:
    if value is None:
        return
    overrides.setdefault(section, {})[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a MOMIA2 settings sweep from JSON.")
    parser.add_argument("--sweep", type=Path, default=Path("configs/settings_sweep.json"))
    parser.add_argument("--output-dir", type=Path, help="Root directory for sweep outputs")
    parser.add_argument("--input-dir", type=Path, help="Override base input directory")
    parser.add_argument("--pattern", type=str, help="Override image glob pattern")
    parser.add_argument("--limit", type=int, help="Override runtime.limit for every sweep run")
    parser.add_argument("--workers", type=int, help="Override runtime.workers for every sweep run")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _load_sweep_settings(sweep_path: Path) -> tuple[Path, list[dict[str, Any]], Path]:
    payload = load_json_file(sweep_path)
    base_settings_path = Path(payload.get("base_settings", "configs/momia2_fullsize.json"))
    runs = payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{sweep_path} must contain a non-empty 'runs' array")
    output_root = Path(payload.get("output_dir", "output/momia2_sweep"))
    return base_settings_path, runs, output_root


def run_sweep(
    *,
    sweep_path: Path,
    output_dir: Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> Path:
    base_settings_path, runs, configured_output_root = _load_sweep_settings(sweep_path)
    output_root = output_dir or configured_output_root
    output_root.mkdir(parents=True, exist_ok=True)

    base_settings = resolve_effective_settings(
        settings_path=base_settings_path,
        allow_missing_settings=False,
    )

    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "sweep_path": str(sweep_path),
        "base_settings": str(base_settings_path),
        "output_dir": str(output_root),
        "runs": [],
    }

    for index, run_cfg in enumerate(runs, start=1):
        if not isinstance(run_cfg, Mapping):
            raise ValueError(f"Sweep run {index} must be an object")
        name = _safe_name(str(run_cfg.get("name", f"run_{index:02d}")))
        run_output_dir = output_root / f"{index:02d}_{name}"
        settings = deep_merge(base_settings, run_cfg.get("overrides", {}))
        settings = deep_merge(settings, {"io": {"output_dir": str(run_output_dir)}})
        if cli_overrides:
            settings = deep_merge(settings, cli_overrides)
        validate_settings(settings)

        logger.info(f"Running sweep case {index}: {name}")
        summary = run_batch(settings=settings, settings_path=base_settings_path)
        row = {
            "run_index": index,
            "run_name": name,
            "output_dir": str(run_output_dir),
            "processed": summary.processed,
            "succeeded": summary.succeeded,
            "failed": summary.failed,
            "skipped": summary.skipped,
            "total_cells": sum(item.n_cells_total or 0 for item in summary.results),
            "filtered_cells": sum(item.n_cells_filtered or 0 for item in summary.results),
            "method": settings["segmentation"]["method"],
            "min_particle_size": settings["segmentation"]["min_particle_size"],
            "min_hole_size": settings["segmentation"]["min_hole_size"],
            "dark_foreground": settings["segmentation"]["dark_foreground"],
            "window_size": settings["segmentation"]["window_size"],
            "k": settings["segmentation"]["k"],
            "area_min": settings["filtering"]["area_min"],
            "area_max": settings["filtering"]["area_max"],
            "length_min": settings["filtering"]["length_min"],
            "length_max": settings["filtering"]["length_max"],
        }
        rows.append(row)
        manifest["runs"].append({"settings": settings, "summary": asdict(summary)})

    summary_path = output_root / "sweep_summary.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(output_root / "sweep_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    return summary_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    _set_if_present(overrides, "io", "input_dir", str(args.input_dir) if args.input_dir else None)
    _set_if_present(overrides, "io", "pattern", args.pattern)
    _set_if_present(overrides, "runtime", "limit", args.limit)
    _set_if_present(overrides, "runtime", "workers", args.workers)
    _set_if_present(overrides, "runtime", "force", args.force)
    _set_if_present(overrides, "runtime", "dry_run", args.dry_run)

    summary_path = run_sweep(
        sweep_path=args.sweep,
        output_dir=args.output_dir,
        cli_overrides=overrides,
    )
    logger.info(f"Wrote sweep summary to {summary_path}")
    return 0
