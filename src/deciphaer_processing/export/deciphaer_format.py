"""
Export functions for deciphaer single-cell format.

Exports feature data as:
- sc_morph_data.csv: Features × Cells matrix
- sc_morph_meta.csv: Cell metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from loguru import logger


def export_deciphaer_format(
    features_list: list[dict[str, Any]],
    cell_ids: list[str],
    metadata_list: list[dict[str, Any]],
    output_dir: str | Path,
    prefix: str = "sc_morph",
) -> tuple[Path, Path, Path]:
    """
    Export features and metadata in deciphaer-preprocessing format.

    Creates three CSV files:
    - {prefix}_data.csv: Features × Cells matrix (feature as first column, cells as remaining)
    - {prefix}_meta.csv: Cell metadata (Cell_ID, drug, GroupLabels_Strict only)
    - {prefix}_mapping.csv: Full traceability mapping to segmentation results

    Args:
        features_list: List of feature dictionaries, one per cell
        cell_ids: List of cell IDs corresponding to features
        metadata_list: List of metadata dictionaries, one per cell
        output_dir: Output directory
        prefix: Prefix for output files (default: "sc_morph")

    Returns:
        Tuple of (data_path, meta_path, mapping_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build data matrix (feature as first column, not index)
    data_df = _build_data_matrix(features_list, cell_ids)
    data_path = output_dir / f"{prefix}_data.csv"
    data_df.to_csv(data_path, index=False)
    logger.info(f"Saved {len(data_df)} features × {len(cell_ids)} cells to {data_path}")

    # Build metadata (minimal: Cell_ID, drug, GroupLabels_Strict)
    meta_df = _build_metadata(cell_ids, metadata_list)
    meta_path = output_dir / f"{prefix}_meta.csv"
    meta_df.to_csv(meta_path, index=False)
    logger.info(f"Saved metadata for {len(meta_df)} cells to {meta_path}")

    # Build mapping table (full traceability)
    mapping_df = _build_mapping_table(cell_ids, metadata_list)
    mapping_path = output_dir / f"{prefix}_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    logger.info(f"Saved mapping table for {len(mapping_df)} cells to {mapping_path}")

    return data_path, meta_path, mapping_path


def _build_data_matrix(
    features_list: list[dict[str, Any]],
    cell_ids: list[str],
) -> pd.DataFrame:
    """
    Build Features × Cells matrix in deciphaer-preprocessing format.

    Args:
        features_list: List of feature dictionaries
        cell_ids: List of cell IDs

    Returns:
        DataFrame with 'feature' as first column and cells as remaining columns
    """
    if len(features_list) == 0:
        return pd.DataFrame()

    # Get all feature names (union across all cells)
    all_features = set()
    for features in features_list:
        all_features.update(features.keys())

    # Sort feature names for consistency
    feature_names = sorted(all_features)

    # Build matrix with 'feature' as first column (not index)
    data = {"feature": feature_names}
    for cell_id, features in zip(cell_ids, features_list):
        data[cell_id] = [features.get(feat, float("nan")) for feat in feature_names]

    df = pd.DataFrame(data)

    return df


def _build_metadata(
    cell_ids: list[str],
    metadata_list: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Build metadata DataFrame in deciphaer-preprocessing format.

    Only includes the 3 columns required by preprocessing pipeline:
    Cell_ID, drug, GroupLabels_Strict

    Args:
        cell_ids: List of cell IDs
        metadata_list: List of metadata dictionaries

    Returns:
        DataFrame with cell metadata (3 columns)
    """
    rows = []
    for cell_id, meta in zip(cell_ids, metadata_list):
        row = {
            "Cell_ID": cell_id,
            "drug": meta.get("drug", "UNK"),
            "GroupLabels_Strict": meta.get("GroupLabels_Strict", "UNK"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _build_mapping_table(
    cell_ids: list[str],
    metadata_list: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Build full traceability mapping table.

    Contains all metadata for tracing back to segmentation results:
    - Cell_ID: Hash-based ID used in preprocessing
    - original_cell_id: {source_filename}_{mask_label}
    - source_path: Full path to source image
    - mask_label: Integer label in segmentation mask
    - plate, scene, position: Parsed from filename
    - drug, GroupLabels_Strict: Treatment assignments

    Args:
        cell_ids: List of cell IDs
        metadata_list: List of metadata dictionaries

    Returns:
        DataFrame with full traceability mapping
    """
    rows = []
    for cell_id, meta in zip(cell_ids, metadata_list):
        row = {
            "Cell_ID": cell_id,
            "original_cell_id": meta.get("original_cell_id", ""),
            "source_path": meta.get("source_path", ""),
            "source_filename": meta.get("source_filename", ""),
            "mask_label": meta.get("mask_label", 0),
            "plate": meta.get("plate", "UNK"),
            "scene": meta.get("scene", "UNK"),
            "position": meta.get("position", "UNK"),
            "drug": meta.get("drug", "UNK"),
            "GroupLabels_Strict": meta.get("GroupLabels_Strict", "UNK"),
            "bbox_min_row": meta.get("bbox_min_row", 0),
            "bbox_min_col": meta.get("bbox_min_col", 0),
            "bbox_max_row": meta.get("bbox_max_row", 0),
            "bbox_max_col": meta.get("bbox_max_col", 0),
        }

        # Add any additional metadata fields
        for key, value in meta.items():
            if key not in row:
                row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def export_features_polars(
    features_list: list[dict[str, Any]],
    cell_ids: list[str],
    output_path: str | Path,
) -> Path:
    """
    Export features as a Polars-friendly Parquet file.

    Features are stored in a tidy format (one row per cell).

    Args:
        features_list: List of feature dictionaries
        cell_ids: List of cell IDs
        output_path: Output Parquet path

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build rows (one per cell)
    rows = []
    for cell_id, features in zip(cell_ids, features_list):
        row = {"cell_id": cell_id}
        row.update(features)
        rows.append(row)

    df = pl.DataFrame(rows)
    df.write_parquet(output_path)
    logger.info(f"Saved features for {len(df)} cells to {output_path}")

    return output_path


def merge_with_existing(
    new_data_path: str | Path,
    new_meta_path: str | Path,
    existing_data_path: str | Path,
    existing_meta_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Merge new data with existing deciphaer format files.

    Args:
        new_data_path: Path to new data CSV
        new_meta_path: Path to new metadata CSV
        existing_data_path: Path to existing data CSV
        existing_meta_path: Path to existing metadata CSV
        output_dir: Output directory for merged files

    Returns:
        Tuple of (merged_data_path, merged_meta_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing data (feature is first column, not index)
    existing_data = pd.read_csv(existing_data_path)
    existing_meta = pd.read_csv(existing_meta_path)

    # Load new data (feature is first column, not index)
    new_data = pd.read_csv(new_data_path)
    new_meta = pd.read_csv(new_meta_path)

    # Check for duplicate cell IDs
    existing_ids = set(existing_meta["Cell_ID"])
    new_ids = set(new_meta["Cell_ID"])
    duplicates = existing_ids & new_ids

    if duplicates:
        logger.warning(f"Found {len(duplicates)} duplicate cell IDs, skipping them")
        # Filter out duplicates from new data
        keep_mask = ~new_meta["Cell_ID"].isin(duplicates)
        new_meta = new_meta[keep_mask]
        # Keep feature column + non-duplicate cell columns
        new_data = new_data[["feature"] + [c for c in new_data.columns if c != "feature" and c not in duplicates]]

    # Merge data (union of features, union of cells)
    # Set feature as index for merging, then reset
    existing_data_indexed = existing_data.set_index("feature")
    new_data_indexed = new_data.set_index("feature")

    all_features = list(set(existing_data_indexed.index) | set(new_data_indexed.index))
    merged_data = pd.DataFrame(index=all_features)

    for col in existing_data_indexed.columns:
        merged_data[col] = existing_data_indexed.reindex(all_features)[col]

    for col in new_data_indexed.columns:
        merged_data[col] = new_data_indexed.reindex(all_features)[col]

    # Reset index to make feature a column again
    merged_data = merged_data.reset_index().rename(columns={"index": "feature"})

    # Merge metadata
    merged_meta = pd.concat([existing_meta, new_meta], ignore_index=True)

    # Save (feature as first column, no index)
    merged_data_path = output_dir / "sc_morph_data_merged.csv"
    merged_meta_path = output_dir / "sc_morph_meta_merged.csv"

    merged_data.to_csv(merged_data_path, index=False)
    merged_meta.to_csv(merged_meta_path, index=False)

    logger.info(f"Merged {len(merged_data.columns)} cells, {len(merged_data)} features")

    return merged_data_path, merged_meta_path


def get_feature_summary(data_path: str | Path) -> pd.DataFrame:
    """
    Generate summary statistics for features.

    Args:
        data_path: Path to sc_morph_data.csv

    Returns:
        DataFrame with feature statistics (mean, std, min, max, etc.)
    """
    data = pd.read_csv(data_path)
    # Set feature column as index for summary computation
    data = data.set_index("feature")

    n_cells = len(data.columns)
    summary = pd.DataFrame({
        "mean": data.mean(axis=1),
        "std": data.std(axis=1),
        "min": data.min(axis=1),
        "max": data.max(axis=1),
        "q25": data.quantile(0.25, axis=1),
        "q50": data.quantile(0.50, axis=1),
        "q75": data.quantile(0.75, axis=1),
        "n_valid": data.notna().sum(axis=1),
        "pct_missing": (data.isna().sum(axis=1) / n_cells * 100).round(2),
    })

    return summary
