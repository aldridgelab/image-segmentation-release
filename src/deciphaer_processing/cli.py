"""
CLI for single-cell feature extraction pipeline.

Usage:
    # Extract from manual masks
    python -m deciphaer_processing.cli extract-manual \
        /path/to/masks \
        /path/to/metadata \
        /path/to/images \
        processing_output/

    # Extract from U-Net predictions
    python -m deciphaer_processing.cli extract-unet \
        images_dir/ \
        tb_unet/checkpoints/focal_loss/best.pt \
        processing_output/

    # Process single image
    python -m deciphaer_processing.cli extract-single \
        image.tif \
        --mask mask.tif \
        --output-dir processing_output/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from deciphaer_processing.pipeline import SingleCellPipeline, PipelineConfig, FilterParams
from deciphaer_processing.padmap.registry import apply_padmap_to_file

app = typer.Typer(
    name="processing",
    help="Single-cell feature extraction pipeline",
    add_completion=False,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_app_config_path() -> Optional[Path]:
    candidate = _repo_root() / "session_output" / "app_config.json"
    return candidate if candidate.exists() else None


def _default_user_settings_path() -> Optional[Path]:
    candidate = _repo_root() / "session_output" / "user_settings.json"
    return candidate if candidate.exists() else None


def _load_app_config(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        import json
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load app config: {exc}")
        return {}


def _apply_user_settings(
    filter_params: FilterParams,
    settings_path: Optional[Path],
    use_app_settings: bool,
) -> FilterParams:
    if not use_app_settings or settings_path is None or not settings_path.exists():
        return filter_params
    data = _load_app_config(settings_path)
    filter_params.area_min = data.get("area_min", filter_params.area_min)
    filter_params.area_max = data.get("area_max", filter_params.area_max)
    filter_params.eccentricity_min = data.get("eccentricity_min", filter_params.eccentricity_min)
    filter_params.eccentricity_max = data.get("eccentricity_max", filter_params.eccentricity_max)
    filter_params.solidity_min = data.get("solidity_min", filter_params.solidity_min)
    filter_params.solidity_max = data.get("solidity_max", filter_params.solidity_max)
    filter_params.aspect_ratio_min = data.get("aspect_ratio_min", filter_params.aspect_ratio_min)
    filter_params.aspect_ratio_max = data.get("aspect_ratio_max", filter_params.aspect_ratio_max)
    filter_params.exclude_edge = data.get("exclude_edge", filter_params.exclude_edge)
    filter_params.edge_distance = data.get("edge_distance", filter_params.edge_distance)
    return filter_params


def _propagate_padmap_labels(
    updated_meta_path: Path,
    outputs: dict[str, Path],
) -> None:
    """Propagate drug/GroupLabels_Strict from the padmapped meta CSV to
    the mapping CSV and mask manifest files.

    The meta CSV is the source of truth after padmap application. Any cell
    not present in it (e.g. dropped by an inner padmap join) is removed
    from every other output file, and its mask TIFF is deleted from disk.
    Discarded cell IDs are logged to ``padmap_discarded_cells.json``.
    """
    import json
    import polars as pl

    meta_df = pl.read_csv(updated_meta_path)
    drug_lookup = meta_df.select("Cell_ID", "drug", "GroupLabels_Strict")
    kept_ids = set(meta_df["Cell_ID"].to_list())

    label_cols = ["drug", "GroupLabels_Strict"]

    files_to_update: list[tuple[str, str]] = [
        ("mapping_csv", "Cell_ID"),
        ("mask_manifest_csv", "cell_id"),
        ("mask_manifest_parquet", "cell_id"),
    ]

    masks_dir: Path | None = None
    removed_cell_ids: set[str] = set()
    removed_mask_filenames: set[str] = set()

    for output_key, id_col in files_to_update:
        path = outputs.get(output_key)
        if path is None or not path.exists():
            continue

        is_parquet = path.suffix == ".parquet"
        df = pl.read_parquet(path) if is_parquet else pl.read_csv(path)

        if id_col not in df.columns:
            continue

        n_before = len(df)

        # Identify rows about to be dropped
        dropped = df.filter(~pl.col(id_col).is_in(kept_ids))
        removed_cell_ids.update(dropped[id_col].to_list())
        if "mask_filename" in dropped.columns:
            removed_mask_filenames.update(dropped["mask_filename"].to_list())
            if masks_dir is None:
                masks_dir = path.parent / "masks"

        # Drop old label columns, inner-join to keep only matched cells
        keep_cols = [c for c in df.columns if c not in label_cols]
        df = df.select(keep_cols).join(
            drug_lookup.rename({"Cell_ID": id_col}),
            on=id_col,
            how="inner",
        )

        if is_parquet:
            df.write_parquet(path)
        else:
            df.write_csv(path)

        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(f"Removed {n_dropped} unmatched cells from {path.name}")
        logger.info(f"Propagated drug labels to {path.name}")

    # Delete orphaned mask TIFFs from disk
    n_deleted = 0
    if removed_mask_filenames and masks_dir is not None and masks_dir.exists():
        for fname in removed_mask_filenames:
            mask_path = masks_dir / fname
            if mask_path.exists():
                mask_path.unlink()
                n_deleted += 1
        if n_deleted:
            logger.info(f"Deleted {n_deleted} orphaned mask files from {masks_dir}")

    # Save a log of all discarded cells for traceability
    if removed_cell_ids:
        output_dir = updated_meta_path.parent
        discarded_log = output_dir / "padmap_discarded_cells.json"
        with open(discarded_log, "w") as f:
            json.dump(
                {
                    "reason": "unmatched_padmap",
                    "count": len(removed_cell_ids),
                    "cell_ids": sorted(removed_cell_ids),
                },
                f,
                indent=2,
            )
        logger.info(
            f"Logged {len(removed_cell_ids)} padmap-discarded cells to {discarded_log}"
        )


def _run_unet_pipeline(
    images_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    area_min: Optional[float],
    area_max: Optional[float],
    eccentricity_min: Optional[float],
    eccentricity_max: Optional[float],
    solidity_min: Optional[float],
    solidity_max: Optional[float],
    exclude_edge: bool,
    edge_distance: int,
    pixel_microns: float,
    skip_centerline: bool,
    skip_intensity: bool,
    cell_prob_min: Optional[float],
    cell_margin: Optional[float],
    image_pattern: Optional[str],
    limit: Optional[int],
    use_app_settings: bool,
    settings_path: Optional[Path],
    app_config_path: Optional[Path],
    prefix: str,
    no_parquet: bool,
    no_csv: bool,
    padmap_path: Path,
    padmap_join: str,
    padmap_id_source: str,
    padmap_id_regex: Optional[str],
    save_masks: bool,
    masks_dir: Optional[Path],
) -> None:
    resolved_settings = settings_path or _default_user_settings_path()
    resolved_app_config = app_config_path or _default_app_config_path()
    app_config = _load_app_config(resolved_app_config) if resolved_app_config else {}

    if cell_prob_min is None:
        cell_prob_min = float(app_config.get("unet_cell_prob_min", 0.6))
    if cell_margin is None:
        cell_margin = float(app_config.get("unet_cell_margin", 0.1))

    filter_params = FilterParams(
        area_min=area_min,
        area_max=area_max,
        eccentricity_min=eccentricity_min,
        eccentricity_max=eccentricity_max,
        solidity_min=solidity_min,
        solidity_max=solidity_max,
        exclude_edge=exclude_edge,
        edge_distance=edge_distance,
    )
    filter_params = _apply_user_settings(filter_params, resolved_settings, use_app_settings)

    config = PipelineConfig(
        pixel_microns=pixel_microns,
        filter_params=filter_params,
        extract_centerline=not skip_centerline,
        extract_intensity=not skip_intensity,
        output_parquet=not no_parquet,
        output_csv=not no_csv,
        unet_cell_prob_min=cell_prob_min,
        unet_cell_margin=cell_margin,
        save_masks=save_masks,
        masks_output_dir=str(masks_dir) if masks_dir else None,
    )

    pipeline = SingleCellPipeline(config=config, output_dir=output_dir)
    stats = pipeline.process_unet_predictions(
        images_dir,
        checkpoint,
        image_pattern=image_pattern,
        limit=limit,
    )
    outputs = pipeline.export(prefix=prefix)

    typer.echo("\n" + "=" * 50)
    typer.echo("Feature Extraction Summary")
    typer.echo("=" * 50)
    typer.echo(f"Total images processed: {stats['total_images']}")
    typer.echo(f"Total cells detected: {stats['total_cells']}")
    typer.echo(f"No cell detected: {stats['no_cell_detected']}")
    typer.echo(f"Cells passed filter: {stats['passed_filter']}")
    typer.echo(f"Cells filtered out: {stats['failed_filter']}")
    typer.echo(f"Errors: {stats['errors']}")
    typer.echo("\nOutput files:")
    for name, path in outputs.items():
        typer.echo(f"  {name}: {path}")

    typer.echo("\n" + "=" * 50)
    typer.echo("Applying Padmap")
    typer.echo("=" * 50)

    meta_path = outputs.get("meta_csv")
    data_path = outputs.get("data_csv")
    mapping_path = outputs.get("mapping_csv")

    if meta_path is None:
        typer.echo("Error: No metadata CSV was generated (--no-csv?)", err=True)
        raise typer.Exit(1)

    try:
        output_file, report, data_output = apply_padmap_to_file(
            meta_path=meta_path,
            padmap_path=padmap_path,
            output_path=None,  # Will create _padmapped suffix
            data_path=data_path if padmap_join == "inner" else None,
            mapping_path=mapping_path,
            id_source=padmap_id_source,  # type: ignore
            id_regex=padmap_id_regex,
            padmap_id_col="id",
            padmap_drug_col="drug",
            join=padmap_join,  # type: ignore
            update_drug=True,
            update_group_labels=True,
            in_place=True,  # Overwrite the original files
        )

        typer.echo(f"Total cells: {report.total_cells}")
        typer.echo(f"Matched cells: {report.matched_cells}")
        typer.echo(f"Unmatched cells: {report.unmatched_cells}")
        typer.echo(f"Match percentage: {report.match_percentage:.1f}%")
        typer.echo(f"Unique drugs matched: {report.unique_drugs_matched}")
        if report.unmatched_ids:
            typer.echo(f"Sample unmatched IDs: {report.unmatched_ids[:5]}")

        # Propagate drug labels to mapping CSV and mask manifest
        _propagate_padmap_labels(output_file, outputs)

    except Exception as e:
        typer.echo(f"Error: Padmap application failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("extract-manual")
def extract_manual(
    masks_dir: Path = typer.Argument(..., help="Directory containing mask TIFFs"),
    metadata_dir: Path = typer.Argument(..., help="Directory containing metadata JSONs"),
    images_dir: Path = typer.Argument(..., help="Directory containing source images"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    # Filter parameters
    area_min: Optional[float] = typer.Option(100.0, help="Minimum area in pixels"),
    area_max: Optional[float] = typer.Option(500.0, help="Maximum area in pixels"),
    eccentricity_max: Optional[float] = typer.Option(None, help="Maximum eccentricity"),
    solidity_min: Optional[float] = typer.Option(None, help="Minimum solidity"),
    exclude_edge: bool = typer.Option(True, help="Exclude cells touching edge"),
    edge_distance: int = typer.Option(8, help="Edge exclusion distance in pixels"),
    # Processing options
    pixel_microns: float = typer.Option(0.10317, help="Pixel size in microns"),
    skip_centerline: bool = typer.Option(False, help="Skip centerline feature extraction"),
    skip_intensity: bool = typer.Option(False, help="Skip intensity feature extraction"),
    # Output options
    prefix: str = typer.Option("sc_morph", help="Output file prefix"),
    no_parquet: bool = typer.Option(False, help="Skip Parquet output"),
    no_csv: bool = typer.Option(False, help="Skip CSV output"),
) -> None:
    """
    Extract features from manually curated masks.

    Reads mask TIFFs and corresponding metadata JSONs, extracts MOMIA2-style
    morphological features, and exports in deciphaer format.
    """
    logger.info(f"Extracting features from manual masks in {masks_dir}")

    # Build configuration
    filter_params = FilterParams(
        area_min=area_min,
        area_max=area_max,
        eccentricity_max=eccentricity_max,
        solidity_min=solidity_min,
        exclude_edge=exclude_edge,
        edge_distance=edge_distance,
    )

    config = PipelineConfig(
        pixel_microns=pixel_microns,
        filter_params=filter_params,
        extract_centerline=not skip_centerline,
        extract_intensity=not skip_intensity,
        output_parquet=not no_parquet,
        output_csv=not no_csv,
    )

    # Run pipeline
    pipeline = SingleCellPipeline(config=config, output_dir=output_dir)
    stats = pipeline.process_manual_masks(masks_dir, metadata_dir, images_dir)
    outputs = pipeline.export(prefix=prefix)

    # Print summary
    typer.echo("\n" + "=" * 50)
    typer.echo("Feature Extraction Summary")
    typer.echo("=" * 50)
    typer.echo(f"Total masks processed: {stats['total_masks']}")
    typer.echo(f"Total cells found: {stats['total_cells']}")
    typer.echo(f"Cells passed filter: {stats['passed_filter']}")
    typer.echo(f"Cells filtered out: {stats['failed_filter']}")
    typer.echo(f"Errors: {stats['errors']}")
    typer.echo("\nOutput files:")
    for name, path in outputs.items():
        typer.echo(f"  {name}: {path}")


@app.command("extract-unet")
def extract_unet(
    images_dir: Path = typer.Argument(..., help="Directory containing source images"),
    checkpoint: Path = typer.Argument(..., help="Path to U-Net checkpoint"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    # Filter parameters
    area_min: Optional[float] = typer.Option(100.0, help="Minimum area in pixels"),
    area_max: Optional[float] = typer.Option(500.0, help="Maximum area in pixels"),
    eccentricity_min: Optional[float] = typer.Option(None, help="Minimum eccentricity"),
    eccentricity_max: Optional[float] = typer.Option(None, help="Maximum eccentricity"),
    solidity_max: Optional[float] = typer.Option(None, help="Maximum solidity"),
    solidity_min: Optional[float] = typer.Option(None, help="Minimum solidity"),
    exclude_edge: bool = typer.Option(True, help="Exclude cells touching edge"),
    edge_distance: int = typer.Option(8, help="Edge exclusion distance in pixels"),
    # Processing options
    pixel_microns: float = typer.Option(0.10317, help="Pixel size in microns"),
    skip_centerline: bool = typer.Option(False, help="Skip centerline feature extraction"),
    skip_intensity: bool = typer.Option(False, help="Skip intensity feature extraction"),
    cell_prob_min: Optional[float] = typer.Option(
        None, help="Minimum P(Cell) for U-Net mask"
    ),
    cell_margin: Optional[float] = typer.Option(
        None, help="Minimum margin between P(Cell) and P(Non-cell)"
    ),
    image_pattern: Optional[str] = typer.Option(
        None, help="Glob pattern for image stacks"
    ),
    limit: Optional[int] = typer.Option(None, help="Limit number of images"),
    use_app_settings: bool = typer.Option(
        False, help="Load filter params from user_settings.json"
    ),
    settings_path: Optional[Path] = typer.Option(
        None, help="Path to user_settings.json"
    ),
    app_config_path: Optional[Path] = typer.Option(
        None, help="Path to app_config.json"
    ),
    # Output options
    prefix: str = typer.Option("sc_morph", help="Output file prefix"),
    no_parquet: bool = typer.Option(False, help="Skip Parquet output"),
    no_csv: bool = typer.Option(False, help="Skip CSV output"),
    # Mask export options
    save_masks: bool = typer.Option(False, help="Save per-cell binary masks to disk"),
    masks_dir: Optional[Path] = typer.Option(
        None, help="Directory for saved masks (default: {output_dir}/masks/)"
    ),
    # Padmap options
    padmap: Path = typer.Option(
        ...,
        "--padmap",
        help="Path to padmap CSV for automatic drug label assignment (required).",
    ),
    padmap_join: str = typer.Option(
        "inner",
        help="Join type for padmap: 'inner' (only matched cells) or 'left' (keep all)",
    ),
    padmap_id_source: str = typer.Option(
        "source_path",
        "--padmap-id-source",
        help="Source for padmap ID: source_path, plate_scene, or original_cell_id",
    ),
    padmap_id_regex: Optional[str] = typer.Option(
        None,
        "--padmap-id-regex",
        help="Regex capture group for source_path/original_cell_id IDs",
    ),
) -> None:
    """
    Extract features using U-Net predictions.

    Runs U-Net inference on each image, then extracts MOMIA2-style
    morphological features and exports in deciphaer format.

    Requires --padmap. Drug labels are automatically applied to the output.
    """
    logger.info(f"Extracting features using U-Net from {images_dir}")

    _run_unet_pipeline(
        images_dir,
        checkpoint,
        output_dir,
        area_min=area_min,
        area_max=area_max,
        eccentricity_min=eccentricity_min,
        eccentricity_max=eccentricity_max,
        solidity_min=solidity_min,
        solidity_max=solidity_max,
        exclude_edge=exclude_edge,
        edge_distance=edge_distance,
        pixel_microns=pixel_microns,
        skip_centerline=skip_centerline,
        skip_intensity=skip_intensity,
        cell_prob_min=cell_prob_min,
        cell_margin=cell_margin,
        image_pattern=image_pattern,
        limit=limit,
        use_app_settings=use_app_settings,
        settings_path=settings_path,
        app_config_path=app_config_path,
        prefix=prefix,
        no_parquet=no_parquet,
        no_csv=no_csv,
        padmap_path=padmap,
        padmap_join=padmap_join,
        padmap_id_source=padmap_id_source,
        padmap_id_regex=padmap_id_regex,
        save_masks=save_masks,
        masks_dir=masks_dir,
    )


@app.command("extract-unet-dir")
def extract_unet_dir(
    images_dir: Path = typer.Argument(..., help="Directory containing source images"),
    output_dir: Path = typer.Option(
        Path("processing_output_unet"),
        "--output-dir",
        "-o",
        help="Output directory",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Path to U-Net checkpoint (defaults to app_config.json if available)",
    ),
    # Filter parameters
    area_min: Optional[float] = typer.Option(100.0, help="Minimum area in pixels"),
    area_max: Optional[float] = typer.Option(500.0, help="Maximum area in pixels"),
    eccentricity_min: Optional[float] = typer.Option(None, help="Minimum eccentricity"),
    eccentricity_max: Optional[float] = typer.Option(None, help="Maximum eccentricity"),
    solidity_min: Optional[float] = typer.Option(None, help="Minimum solidity"),
    solidity_max: Optional[float] = typer.Option(None, help="Maximum solidity"),
    exclude_edge: bool = typer.Option(True, help="Exclude cells touching edge"),
    edge_distance: int = typer.Option(8, help="Edge exclusion distance in pixels"),
    # Processing options
    pixel_microns: float = typer.Option(0.10317, help="Pixel size in microns"),
    skip_centerline: bool = typer.Option(False, help="Skip centerline feature extraction"),
    skip_intensity: bool = typer.Option(False, help="Skip intensity feature extraction"),
    cell_prob_min: Optional[float] = typer.Option(
        None, help="Minimum P(Cell) for U-Net mask"
    ),
    cell_margin: Optional[float] = typer.Option(
        None, help="Minimum margin between P(Cell) and P(Non-cell)"
    ),
    image_pattern: Optional[str] = typer.Option(
        None, help="Glob pattern for image stacks"
    ),
    limit: Optional[int] = typer.Option(None, help="Limit number of images"),
    use_app_settings: bool = typer.Option(
        True, help="Load filter params from user_settings.json"
    ),
    settings_path: Optional[Path] = typer.Option(
        None, help="Path to user_settings.json"
    ),
    app_config_path: Optional[Path] = typer.Option(
        None, help="Path to app_config.json"
    ),
    # Output options
    prefix: str = typer.Option("sc_morph", help="Output file prefix"),
    no_parquet: bool = typer.Option(False, help="Skip Parquet output"),
    no_csv: bool = typer.Option(False, help="Skip CSV output"),
    # Mask export options
    save_masks: bool = typer.Option(False, help="Save per-cell binary masks to disk"),
    masks_dir: Optional[Path] = typer.Option(
        None, help="Directory for saved masks (default: {output_dir}/masks/)"
    ),
    # Padmap options
    padmap: Path = typer.Option(
        ...,
        "--padmap",
        help="Path to padmap CSV for automatic drug label assignment (required).",
    ),
    padmap_join: str = typer.Option(
        "inner",
        help="Join type for padmap: 'inner' (only matched cells) or 'left' (keep all)",
    ),
    padmap_id_source: str = typer.Option(
        "source_path",
        "--padmap-id-source",
        help="Source for padmap ID: source_path, plate_scene, or original_cell_id",
    ),
    padmap_id_regex: Optional[str] = typer.Option(
        None,
        "--padmap-id-regex",
        help="Regex capture group for source_path/original_cell_id IDs",
    ),
) -> None:
    """
    Extract features using U-Net predictions (simplified interface).

    Requires --padmap. Drug labels are automatically applied to the output.
    """
    logger.info(f"Extracting features using U-Net from {images_dir}")

    resolved_app_config = app_config_path or _default_app_config_path()
    app_config = _load_app_config(resolved_app_config) if resolved_app_config else {}
    resolved_checkpoint = checkpoint or app_config.get("unet_checkpoint")
    if resolved_checkpoint is None:
        typer.echo("Error: checkpoint path required (set --checkpoint or app_config.json)", err=True)
        raise typer.Exit(1)

    _run_unet_pipeline(
        images_dir,
        Path(resolved_checkpoint),
        output_dir,
        area_min=area_min,
        area_max=area_max,
        eccentricity_min=eccentricity_min,
        eccentricity_max=eccentricity_max,
        solidity_min=solidity_min,
        solidity_max=solidity_max,
        exclude_edge=exclude_edge,
        edge_distance=edge_distance,
        pixel_microns=pixel_microns,
        skip_centerline=skip_centerline,
        skip_intensity=skip_intensity,
        cell_prob_min=cell_prob_min,
        cell_margin=cell_margin,
        image_pattern=image_pattern,
        limit=limit,
        use_app_settings=use_app_settings,
        settings_path=settings_path,
        app_config_path=app_config_path,
        prefix=prefix,
        no_parquet=no_parquet,
        no_csv=no_csv,
        padmap_path=padmap,
        padmap_join=padmap_join,
        padmap_id_source=padmap_id_source,
        padmap_id_regex=padmap_id_regex,
        save_masks=save_masks,
        masks_dir=masks_dir,
    )


@app.command("extract-single")
def extract_single(
    image_path: Path = typer.Argument(..., help="Path to source image"),
    mask: Optional[Path] = typer.Option(None, help="Path to pre-computed mask"),
    checkpoint: Optional[Path] = typer.Option(None, help="Path to U-Net checkpoint"),
    output_dir: Path = typer.Option(".", help="Output directory"),
    # Processing options
    pixel_microns: float = typer.Option(0.10317, help="Pixel size in microns"),
    json_output: bool = typer.Option(False, help="Output features as JSON"),
) -> None:
    """
    Extract features from a single image.

    Requires either a pre-computed mask (--mask) or a U-Net checkpoint (--checkpoint).
    """
    if mask is None and checkpoint is None:
        typer.echo("Error: Either --mask or --checkpoint must be provided", err=True)
        raise typer.Exit(1)

    logger.info(f"Processing single image: {image_path}")

    config = PipelineConfig(pixel_microns=pixel_microns)
    pipeline = SingleCellPipeline(config=config, output_dir=output_dir)

    features = pipeline.process_single_image(
        image_path,
        mask_path=mask,
        checkpoint_path=checkpoint,
    )

    if features is None:
        typer.echo("No valid cell found in image", err=True)
        raise typer.Exit(1)

    if json_output:
        import json
        typer.echo(json.dumps(features, indent=2, default=str))
    else:
        typer.echo("\nExtracted Features:")
        typer.echo("-" * 40)
        for key, value in sorted(features.items()):
            if isinstance(value, float):
                typer.echo(f"  {key}: {value:.4f}")
            else:
                typer.echo(f"  {key}: {value}")


@app.command("dehash")
def dehash_cell(
    cell_id: str = typer.Argument(..., help="Cell ID to look up"),
    lookup_path: Path = typer.Argument(..., help="Path to cell_lookup.parquet"),
) -> None:
    """
    Look up source information for a cell ID.
    """
    from deciphaer_processing.hashing.lookup import CellLookupTable

    lookup = CellLookupTable(lookup_path)
    info = lookup.dehash_cell(cell_id)

    if info is None:
        typer.echo(f"Cell ID not found: {cell_id}", err=True)
        raise typer.Exit(1)

    typer.echo("\nCell Information:")
    typer.echo("-" * 40)
    for key, value in info.items():
        typer.echo(f"  {key}: {value}")


@app.command("list-features")
def list_features(
    data_path: Path = typer.Argument(..., help="Path to sc_morph_data.csv"),
) -> None:
    """
    List all features in a data file with summary statistics.
    """
    from deciphaer_processing.export.deciphaer_format import get_feature_summary

    summary = get_feature_summary(data_path)

    typer.echo(f"\nFeatures in {data_path.name}:")
    typer.echo("-" * 80)
    typer.echo(f"{'Feature':<40} {'Mean':>10} {'Std':>10} {'Missing%':>10}")
    typer.echo("-" * 80)

    for feature in summary.index:
        row = summary.loc[feature]
        typer.echo(
            f"{feature:<40} {row['mean']:>10.3f} {row['std']:>10.3f} {row['pct_missing']:>10.1f}%"
        )

    typer.echo("-" * 80)
    typer.echo(f"Total features: {len(summary)}")


@app.command("merge")
def merge_datasets(
    data1: Path = typer.Argument(..., help="First sc_morph_data.csv"),
    meta1: Path = typer.Argument(..., help="First sc_morph_meta.csv"),
    data2: Path = typer.Argument(..., help="Second sc_morph_data.csv"),
    meta2: Path = typer.Argument(..., help="Second sc_morph_meta.csv"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
) -> None:
    """
    Merge two deciphaer format datasets.
    """
    from deciphaer_processing.export.deciphaer_format import merge_with_existing

    data_path, meta_path = merge_with_existing(
        data2, meta2, data1, meta1, output_dir
    )

    typer.echo(f"\nMerged files created:")
    typer.echo(f"  Data: {data_path}")
    typer.echo(f"  Metadata: {meta_path}")


@app.command("apply-padmap")
def apply_padmap_cmd(
    meta: Path = typer.Option(
        ...,
        "--meta",
        "-m",
        help="Path to metadata CSV (e.g., sc_morph_meta.csv)",
    ),
    padmap: Path = typer.Option(
        ...,
        "--padmap",
        "-p",
        help="Path to padmap CSV with drug/ID mappings",
    ),
    data: Optional[Path] = typer.Option(
        None,
        "--data",
        "-d",
        help="Path to data CSV (e.g., sc_morph_data.csv). If provided, filters to match metadata.",
    ),
    mapping: Optional[Path] = typer.Option(
        None,
        "--mapping",
        help="Path to mapping CSV for ID derivation (auto-detected if not provided)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: same as input)",
    ),
    id_source: str = typer.Option(
        "plate_scene",
        "--id-source",
        help="Source for padmap ID: plate_scene, source_path, or original_cell_id",
    ),
    id_regex: Optional[str] = typer.Option(
        None,
        "--id-regex",
        help="Regex pattern for extracting ID (used with source_path/original_cell_id)",
    ),
    padmap_id_col: str = typer.Option(
        "id",
        "--padmap-id-col",
        help="Column name for ID in padmap file",
    ),
    padmap_drug_col: str = typer.Option(
        "drug",
        "--padmap-drug-col",
        help="Column name for drug in padmap file",
    ),
    join: str = typer.Option(
        "inner",
        "--join",
        help="Join type: inner (only matched cells), left (keep all), or outer (include unmatched padmap)",
    ),
    update_group_labels: bool = typer.Option(
        True,
        "--update-group-labels/--no-update-group-labels",
        help="Mirror drug to GroupLabels_Strict",
    ),
    in_place: bool = typer.Option(
        False,
        "--in-place",
        help="Overwrite input file instead of creating _padmapped.csv",
    ),
) -> None:
    """
    Apply padmap drug labels to cell metadata.

    Joins drug labels from a padmap CSV to existing cell metadata based on
    plate/scene identifiers. This is a post-processing step that enriches
    feature outputs with drug labels without re-running segmentation.

    The default mode (plate_scene) builds a join key from plate and scene
    columns as "{plate}_s{scene}" (e.g., "0101_s01").

    Examples:

        # Basic usage with default settings
        python -m deciphaer_processing apply-padmap \\
            --meta processing_output_unet/sc_morph_meta.csv \\
            --padmap /path/to/padmap.csv

        # Extract ID from source_path with custom regex
        python -m deciphaer_processing apply-padmap \\
            --meta processing_output_unet/sc_morph_meta.csv \\
            --padmap /path/to/padmap.csv \\
            --id-source source_path \\
            --id-regex "(\\d{4}_s\\d+)"

        # Overwrite original file
        python -m deciphaer_processing apply-padmap \\
            --meta processing_output_unet/sc_morph_meta.csv \\
            --padmap /path/to/padmap.csv \\
            --in-place
    """
    from typing import Literal

    # Validate id_source
    valid_sources = ("plate_scene", "source_path", "original_cell_id")
    if id_source not in valid_sources:
        typer.echo(f"Error: --id-source must be one of {valid_sources}", err=True)
        raise typer.Exit(1)

    # Validate join type
    if join not in ("left", "outer", "inner"):
        typer.echo("Error: --join must be 'inner', 'left', or 'outer'", err=True)
        raise typer.Exit(1)

    # Determine output path
    if output_dir:
        output_path = output_dir / meta.name
    else:
        output_path = None  # Will use default (same dir, _padmapped suffix)

    try:
        output_file, report, data_output = apply_padmap_to_file(
            meta_path=meta,
            padmap_path=padmap,
            output_path=output_path,
            data_path=data,
            mapping_path=mapping,
            id_source=id_source,  # type: ignore
            id_regex=id_regex,
            padmap_id_col=padmap_id_col,
            padmap_drug_col=padmap_drug_col,
            join=join,  # type: ignore
            update_drug=True,
            update_group_labels=update_group_labels,
            in_place=in_place,
        )

        typer.echo("\n" + "=" * 50)
        typer.echo("Padmap Registration Summary")
        typer.echo("=" * 50)
        typer.echo(f"Total cells: {report.total_cells}")
        typer.echo(f"Matched cells: {report.matched_cells}")
        typer.echo(f"Unmatched cells: {report.unmatched_cells}")
        typer.echo(f"Match percentage: {report.match_percentage:.1f}%")
        typer.echo(f"Unique drugs matched: {report.unique_drugs_matched}")
        if report.unmatched_ids:
            typer.echo(f"Sample unmatched IDs: {report.unmatched_ids[:5]}")
        typer.echo(f"\nOutput files:")
        typer.echo(f"  Metadata: {output_file}")
        if data_output:
            typer.echo(f"  Data: {data_output}")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
