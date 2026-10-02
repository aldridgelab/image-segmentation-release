"""Command-line interface for full-size TIFF MOMIA2 batch segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from .batch import DEFAULT_SETTINGS_FILENAME, resolve_effective_settings, run_batch


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    target = payload
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    target[keys[-1]] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MOMIA2 segmentation and feature extraction on full-size TIFF images."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(DEFAULT_SETTINGS_FILENAME),
        help=f"Path to settings JSON (default: {DEFAULT_SETTINGS_FILENAME})",
    )
    parser.add_argument(
        "--allow-missing-settings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow missing --settings and run with built-in defaults plus CLI overrides",
    )
    parser.add_argument(
        "--write-settings-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write the effective settings JSON when --settings does not exist",
    )

    parser.add_argument("--input-dir", type=Path, help="Directory containing TIFF images")
    parser.add_argument("--output-dir", type=Path, help="Output directory root")
    parser.add_argument("--pattern", type=str, help="Glob pattern for image discovery")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--limit", type=int, help="Maximum number of images to process")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log-level", type=str, help="Log level")
    parser.add_argument("--suppress-warnings", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument(
        "--method",
        choices=["isodata", "sauvola", "local", "composite"],
        help="MOMIA2 thresholding method",
    )
    parser.add_argument("--min-particle-size", type=int)
    parser.add_argument("--min-hole-size", type=int)
    parser.add_argument("--dark-foreground", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--window-size", type=int, help="Odd window size for sauvola/local methods")
    parser.add_argument("--k", type=float, help="Sauvola k parameter")

    parser.add_argument("--pixel-microns", type=float)
    parser.add_argument("--channel-names", nargs="+")
    parser.add_argument("--channel-axis", choices=["auto", "first", "last", "none"])
    parser.add_argument("--ref-channel", type=str)
    parser.add_argument("--correct-xy-drift", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-drift", type=float)

    parser.add_argument("--exclude-edge", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--edge-distance", type=int)
    parser.add_argument("--area-min", type=float)
    parser.add_argument("--area-max", type=float)
    parser.add_argument("--eccentricity-min", type=float)
    parser.add_argument("--eccentricity-max", type=float)
    parser.add_argument("--solidity-min", type=float)
    parser.add_argument("--solidity-max", type=float)
    parser.add_argument("--length-min", type=float)
    parser.add_argument("--length-max", type=float)
    parser.add_argument("--width-min", type=float)
    parser.add_argument("--width-max", type=float)
    parser.add_argument("--aspect-ratio-min", type=float)
    parser.add_argument("--aspect-ratio-max", type=float)

    parser.add_argument("--centrality-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--center-window-px", type=int)
    parser.add_argument("--tie-breaker", choices=["area", "eccentricity", "solidity"])
    parser.add_argument("--fail-if-center-cell-filtered", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--emit-ucid", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ucid-prefix", type=str)
    parser.add_argument("--embed-effective-settings", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--write-preview", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--preview-max-size", type=int)
    parser.add_argument("--write-summary-csv", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write-cell-table", action=argparse.BooleanOptionalAction, default=None)
    return parser


def cli_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    scalar_map: dict[str, str] = {
        "input_dir": "io.input_dir",
        "output_dir": "io.output_dir",
        "pattern": "io.pattern",
        "workers": "runtime.workers",
        "limit": "runtime.limit",
        "log_level": "runtime.log_level",
        "method": "segmentation.method",
        "min_particle_size": "segmentation.min_particle_size",
        "min_hole_size": "segmentation.min_hole_size",
        "window_size": "segmentation.window_size",
        "k": "segmentation.k",
        "pixel_microns": "image.pixel_microns",
        "channel_axis": "image.channel_axis",
        "ref_channel": "image.ref_channel",
        "max_drift": "image.max_drift",
        "edge_distance": "filtering.edge_distance",
        "area_min": "filtering.area_min",
        "area_max": "filtering.area_max",
        "eccentricity_min": "filtering.eccentricity_min",
        "eccentricity_max": "filtering.eccentricity_max",
        "solidity_min": "filtering.solidity_min",
        "solidity_max": "filtering.solidity_max",
        "length_min": "filtering.length_min",
        "length_max": "filtering.length_max",
        "width_min": "filtering.width_min",
        "width_max": "filtering.width_max",
        "aspect_ratio_min": "filtering.aspect_ratio_min",
        "aspect_ratio_max": "filtering.aspect_ratio_max",
        "center_window_px": "centrality.center_window_px",
        "tie_breaker": "centrality.tie_breaker",
        "ucid_prefix": "metadata.ucid_prefix",
        "preview_max_size": "outputs.preview_max_size",
    }
    for arg_key, dotted_key in scalar_map.items():
        value = getattr(args, arg_key)
        if value is None:
            continue
        _set_nested(overrides, dotted_key, str(value) if isinstance(value, Path) else value)

    if args.channel_names is not None:
        _set_nested(overrides, "image.channel_names", args.channel_names)

    bool_map: dict[str, str] = {
        "recursive": "io.recursive",
        "skip_existing": "runtime.skip_existing",
        "force": "runtime.force",
        "fail_fast": "runtime.fail_fast",
        "dry_run": "runtime.dry_run",
        "suppress_warnings": "runtime.suppress_warnings",
        "dark_foreground": "segmentation.dark_foreground",
        "correct_xy_drift": "image.correct_xy_drift",
        "exclude_edge": "filtering.exclude_edge",
        "centrality_enabled": "centrality.enabled",
        "fail_if_center_cell_filtered": "centrality.fail_if_center_cell_filtered",
        "emit_ucid": "metadata.emit_ucid",
        "embed_effective_settings": "metadata.embed_effective_settings",
        "write_preview": "outputs.write_preview",
        "write_summary_csv": "outputs.write_summary_csv",
        "write_cell_table": "outputs.write_cell_table",
    }
    for arg_key, dotted_key in bool_map.items():
        value = getattr(args, arg_key)
        if value is not None:
            _set_nested(overrides, dotted_key, bool(value))
    return overrides


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging("INFO")

    settings = resolve_effective_settings(
        settings_path=args.settings,
        cli_overrides=cli_overrides_from_args(args),
        allow_missing_settings=bool(args.allow_missing_settings),
        write_settings_if_missing=bool(args.write_settings_if_missing),
    )
    _configure_logging(str(settings.get("runtime", {}).get("log_level", "INFO")))

    summary = run_batch(settings=settings, settings_path=args.settings)
    logger.info(
        "Processed={processed}, success={success}, failed={failed}, skipped={skipped}, dry_run={dry_run}".format(
            processed=summary.processed,
            success=summary.succeeded,
            failed=summary.failed,
            skipped=summary.skipped,
            dry_run=summary.dry_run,
        )
    )
    return 1 if summary.failed > 0 else 0
