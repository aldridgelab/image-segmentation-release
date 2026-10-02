"""
Padmap registration utilities for mapping drug labels to cell metadata.

This module provides functions to join drug/treatment labels from a padmap CSV
to existing cell metadata, enabling post-hoc enrichment of extracted features
without re-running segmentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
from loguru import logger


DEFAULT_SOURCE_ID_REGEX = r"((?:\d{4}_s\d+)|(?:t\d+_rep\d+_s\d+))"


@dataclass
class PadmapMatchReport:
    """Report summarizing padmap join results."""

    total_cells: int
    matched_cells: int
    unmatched_cells: int
    match_percentage: float
    unique_drugs_matched: int
    unmatched_ids: list[str]

    def __str__(self) -> str:
        return (
            f"PadmapMatchReport(\n"
            f"  total_cells={self.total_cells},\n"
            f"  matched_cells={self.matched_cells},\n"
            f"  unmatched_cells={self.unmatched_cells},\n"
            f"  match_percentage={self.match_percentage:.1f}%,\n"
            f"  unique_drugs_matched={self.unique_drugs_matched},\n"
            f"  unmatched_ids_sample={self.unmatched_ids[:5]}\n"
            f")"
        )

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "total_cells": self.total_cells,
            "matched_cells": self.matched_cells,
            "unmatched_cells": self.unmatched_cells,
            "match_percentage": self.match_percentage,
            "unique_drugs_matched": self.unique_drugs_matched,
            "unmatched_ids": self.unmatched_ids,
        }


def load_padmap(
    path: str | Path,
    id_col: str = "id",
    drug_col: str = "drug",
) -> pl.DataFrame:
    """
    Load a padmap CSV file.

    Args:
        path: Path to the padmap CSV file.
        id_col: Column name for the padmap ID (default: "id").
        drug_col: Column name for the drug label (default: "drug").

    Returns:
        Polars DataFrame with columns renamed to 'padmap_id' and 'padmap_drug'.

    Raises:
        FileNotFoundError: If the padmap file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Padmap file not found: {path}")

    logger.info(f"Loading padmap from {path}")
    df = pl.read_csv(path)

    # Validate columns
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in padmap. Available: {df.columns}")
    if drug_col not in df.columns:
        raise ValueError(f"Drug column '{drug_col}' not found in padmap. Available: {df.columns}")

    # Rename to standard names for joining
    df = df.rename({id_col: "padmap_id", drug_col: "padmap_drug"})

    # Keep only the columns we need
    df = df.select(["padmap_id", "padmap_drug"])

    logger.info(f"Loaded padmap with {len(df)} entries, {df['padmap_drug'].n_unique()} unique drugs")

    return df


def derive_padmap_id(
    meta_df: pl.DataFrame,
    source: Literal["plate_scene", "source_path", "original_cell_id"] = "plate_scene",
    regex: str | None = None,
) -> pl.DataFrame:
    """
    Derive padmap join key from metadata columns.

    Args:
        meta_df: Metadata DataFrame (polars).
        source: Source for deriving the padmap ID:
            - "plate_scene": Combine plate and scene columns as "{plate}_s{scene}"
            - "source_path": Extract from source_path using regex
            - "original_cell_id": Extract from original_cell_id using regex
        regex: Regex pattern with capture group for source_path/original_cell_id modes.
               Default pattern matches "{plate}_s{scene}" and dose-time
               "{time}_rep{rep}_s{scene}" IDs such as "0101_s01" or "t4_rep1_s01".

    Returns:
        DataFrame with added 'padmap_id' column.

    Raises:
        ValueError: If required columns are missing or regex fails.
    """
    logger.info(f"Deriving padmap_id from source='{source}'")

    if source == "plate_scene":
        # Check required columns
        if "plate" not in meta_df.columns or "scene" not in meta_df.columns:
            raise ValueError(
                f"plate_scene mode requires 'plate' and 'scene' columns. "
                f"Available: {meta_df.columns}"
            )

        # Build padmap_id as "{plate}_s{scene}" with zero-padded plate (4 digits) and scene (2 digits)
        # This handles cases where leading zeros may have been stripped during CSV read
        meta_df = meta_df.with_columns(
            pl.concat_str([
                pl.col("plate").cast(pl.Utf8).str.zfill(4),
                pl.lit("_s"),
                pl.col("scene").cast(pl.Utf8).str.zfill(2),
            ]).alias("padmap_id")
        )

    elif source in ("source_path", "original_cell_id"):
        if source not in meta_df.columns:
            raise ValueError(f"Column '{source}' not found. Available: {meta_df.columns}")

        # Default regex pattern for source-derived padmap IDs.
        pattern = regex or DEFAULT_SOURCE_ID_REGEX

        # Extract using regex
        meta_df = meta_df.with_columns(
            pl.col(source).str.extract(pattern, group_index=1).alias("padmap_id")
        )

        # Check for extraction failures
        null_count = meta_df.filter(pl.col("padmap_id").is_null()).height
        if null_count > 0:
            logger.warning(
                f"{null_count} rows had null padmap_id after regex extraction. "
                f"Pattern: {pattern}, source: {source}"
            )

    else:
        raise ValueError(f"Unknown source: {source}. Must be 'plate_scene', 'source_path', or 'original_cell_id'")

    logger.info(f"Derived {meta_df['padmap_id'].n_unique()} unique padmap IDs")

    return meta_df


def apply_padmap(
    meta_df: pl.DataFrame,
    padmap_df: pl.DataFrame,
    join: Literal["left", "outer", "inner"] = "left",
    update_drug: bool = True,
    update_group_labels: bool = True,
) -> tuple[pl.DataFrame, PadmapMatchReport]:
    """
    Apply padmap to metadata by joining on padmap_id.

    Args:
        meta_df: Metadata DataFrame with 'padmap_id' column.
        padmap_df: Padmap DataFrame with 'padmap_id' and 'padmap_drug' columns.
        join: Join type:
            - "left": Keep all cells (unmatched get UNK)
            - "outer": Include unmatched padmap entries
            - "inner": Only keep cells with padmap match (filters out unmatched)
        update_drug: If True, update the 'drug' column with padmap values.
        update_group_labels: If True, update 'GroupLabels_Strict' to mirror 'drug'.

    Returns:
        Tuple of (updated_meta_df, match_report).

    Raises:
        ValueError: If 'padmap_id' column is missing from meta_df.
    """
    if "padmap_id" not in meta_df.columns:
        raise ValueError(
            "meta_df must have 'padmap_id' column. "
            "Call derive_padmap_id() first."
        )

    total_cells = len(meta_df)

    # Perform join
    if join == "left":
        result_df = meta_df.join(padmap_df, on="padmap_id", how="left")
    elif join == "inner":
        result_df = meta_df.join(padmap_df, on="padmap_id", how="inner")
    else:
        result_df = meta_df.join(padmap_df, on="padmap_id", how="outer")

    # Calculate match statistics
    matched_mask = result_df["padmap_drug"].is_not_null()
    matched_cells = result_df.filter(matched_mask).height
    unmatched_cells = total_cells - matched_cells
    match_percentage = (matched_cells / total_cells * 100) if total_cells > 0 else 0.0

    # Get unmatched IDs
    unmatched_ids = (
        result_df
        .filter(~matched_mask)
        .select("padmap_id")
        .unique()
        .to_series()
        .to_list()
    )

    # Count unique drugs matched
    unique_drugs = (
        result_df
        .filter(matched_mask)
        .select("padmap_drug")
        .unique()
        .height
    )

    # Update drug column if requested
    if update_drug:
        if "drug" in result_df.columns:
            # Replace drug with padmap_drug where available, keep original otherwise
            result_df = result_df.with_columns(
                pl.when(pl.col("padmap_drug").is_not_null())
                .then(pl.col("padmap_drug"))
                .otherwise(pl.col("drug"))
                .alias("drug")
            )
        else:
            # Create drug column from padmap_drug
            result_df = result_df.with_columns(
                pl.col("padmap_drug").fill_null("UNK").alias("drug")
            )

    # Update GroupLabels_Strict to mirror drug if requested
    if update_group_labels:
        if "GroupLabels_Strict" in result_df.columns:
            result_df = result_df.with_columns(
                pl.col("drug").alias("GroupLabels_Strict")
            )

    # Drop temporary columns
    columns_to_drop = ["padmap_drug"]
    if "padmap_id" in result_df.columns:
        columns_to_drop.append("padmap_id")
    result_df = result_df.drop([c for c in columns_to_drop if c in result_df.columns])

    # Build report
    report = PadmapMatchReport(
        total_cells=total_cells,
        matched_cells=matched_cells,
        unmatched_cells=unmatched_cells,
        match_percentage=match_percentage,
        unique_drugs_matched=unique_drugs,
        unmatched_ids=unmatched_ids,
    )

    logger.info(f"Padmap applied: {matched_cells}/{total_cells} cells matched ({match_percentage:.1f}%)")

    return result_df, report


def apply_padmap_to_file(
    meta_path: str | Path,
    padmap_path: str | Path,
    output_path: str | Path | None = None,
    data_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
    id_source: Literal["plate_scene", "source_path", "original_cell_id"] = "plate_scene",
    id_regex: str | None = None,
    padmap_id_col: str = "id",
    padmap_drug_col: str = "drug",
    join: Literal["left", "outer", "inner"] = "left",
    update_drug: bool = True,
    update_group_labels: bool = True,
    in_place: bool = False,
) -> tuple[Path, PadmapMatchReport, Path | None]:
    """
    Apply padmap registration to a metadata CSV file.

    Convenience function that combines load, derive, and apply steps.

    If the metadata file doesn't have required columns (plate, scene, etc.),
    automatically looks for a corresponding mapping file to derive padmap IDs.

    Args:
        meta_path: Path to input metadata CSV (e.g., sc_morph_meta.csv).
        padmap_path: Path to padmap CSV.
        output_path: Output path for updated metadata. If None, creates
                     "{input_stem}_padmapped.csv" in same directory.
        data_path: Optional path to data CSV (e.g., sc_morph_data.csv).
                   If provided and join="inner", filters data to match metadata.
        mapping_path: Optional path to mapping CSV for ID derivation.
                      If None, auto-detected as "{meta_stem}_mapping.csv" or
                      "sc_morph_mapping.csv" in same directory.
        id_source: Source for deriving padmap ID.
        id_regex: Regex pattern for source_path/original_cell_id modes.
        padmap_id_col: Column name for ID in padmap file.
        padmap_drug_col: Column name for drug in padmap file.
        join: Join type ("left", "outer", or "inner").
        update_drug: Update 'drug' column with padmap values.
        update_group_labels: Update 'GroupLabels_Strict' to mirror 'drug'.
        in_place: If True, overwrite input file instead of creating new one.

    Returns:
        Tuple of (meta_output_path, match_report, data_output_path or None).
    """
    meta_path = Path(meta_path)
    padmap_path = Path(padmap_path)

    # Load metadata
    logger.info(f"Loading metadata from {meta_path}")
    meta_df = pl.read_csv(meta_path)
    logger.info(f"Loaded {len(meta_df)} cells")

    # Load padmap
    padmap_df = load_padmap(padmap_path, id_col=padmap_id_col, drug_col=padmap_drug_col)

    # Check if we need to use a mapping file for ID derivation
    required_cols = {
        "plate_scene": ["plate", "scene"],
        "source_path": ["source_path"],
        "original_cell_id": ["original_cell_id"],
    }
    needed_cols = required_cols.get(id_source, [])
    missing_cols = [c for c in needed_cols if c not in meta_df.columns]

    mapping_df: pl.DataFrame | None = None
    if missing_cols:
        # Try to find and load mapping file
        if mapping_path:
            resolved_mapping = Path(mapping_path)
        else:
            # Auto-detect mapping file
            # Try: sc_morph_mapping.csv or {meta_stem with _meta replaced by _mapping}.csv
            meta_stem = meta_path.stem
            if meta_stem.endswith("_meta"):
                mapping_stem = meta_stem.replace("_meta", "_mapping")
            else:
                mapping_stem = f"{meta_stem}_mapping"
            resolved_mapping = meta_path.parent / f"{mapping_stem}.csv"

            # Also try generic sc_morph_mapping.csv
            if not resolved_mapping.exists():
                resolved_mapping = meta_path.parent / "sc_morph_mapping.csv"

        if resolved_mapping.exists():
            logger.info(f"Meta file missing columns {missing_cols}, using mapping file: {resolved_mapping}")
            mapping_df = pl.read_csv(resolved_mapping)

            # Verify mapping file has required columns
            mapping_missing = [c for c in needed_cols if c not in mapping_df.columns]
            if mapping_missing:
                raise ValueError(
                    f"Mapping file also missing columns {mapping_missing}. "
                    f"Available: {mapping_df.columns}"
                )

            # Verify Cell_ID columns match
            meta_cell_ids = set(meta_df["Cell_ID"].to_list())
            mapping_cell_ids = set(mapping_df["Cell_ID"].to_list())
            if meta_cell_ids != mapping_cell_ids:
                extra_in_meta = meta_cell_ids - mapping_cell_ids
                extra_in_mapping = mapping_cell_ids - meta_cell_ids
                if extra_in_meta:
                    logger.warning(f"{len(extra_in_meta)} cells in meta not in mapping")
                if extra_in_mapping:
                    logger.warning(f"{len(extra_in_mapping)} cells in mapping not in meta")

            # Join mapping columns to meta for ID derivation
            cols_to_join = ["Cell_ID"] + needed_cols
            meta_df = meta_df.join(
                mapping_df.select(cols_to_join),
                on="Cell_ID",
                how="left"
            )
        else:
            raise ValueError(
                f"Meta file missing columns {missing_cols} for id_source='{id_source}' "
                f"and no mapping file found at {resolved_mapping}"
            )

    # Derive padmap ID
    meta_df = derive_padmap_id(meta_df, source=id_source, regex=id_regex)

    # Apply padmap
    result_df, report = apply_padmap(
        meta_df,
        padmap_df,
        join=join,
        update_drug=update_drug,
        update_group_labels=update_group_labels,
    )

    # Drop columns that were temporarily joined from mapping file
    # Keep only: Cell_ID, drug, GroupLabels_Strict (the minimal preprocessing format)
    cols_to_keep = ["Cell_ID", "drug", "GroupLabels_Strict"]
    extra_cols = [c for c in result_df.columns if c not in cols_to_keep]
    if extra_cols:
        logger.debug(f"Dropping extra columns from output: {extra_cols}")
        result_df = result_df.select(cols_to_keep)

    # Determine output path
    if in_place:
        final_output = meta_path
    elif output_path:
        final_output = Path(output_path)
    else:
        final_output = meta_path.parent / f"{meta_path.stem}_padmapped.csv"

    # Save result
    final_output.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_csv(final_output)
    logger.info(f"Saved updated metadata to {final_output}")

    # Filter data file if provided
    data_output: Path | None = None
    if data_path is not None:
        data_path = Path(data_path)
        logger.info(f"Loading data from {data_path}")

        # Get the cell IDs that remain after filtering
        # Check for Cell_ID or cell_id column
        cell_id_col = "Cell_ID" if "Cell_ID" in result_df.columns else "cell_id"
        kept_cell_ids = set(result_df[cell_id_col].to_list())

        # Load data file - it has features as rows, cells as columns
        # First row is header with cell IDs
        data_df = pl.read_csv(data_path)

        # The first column is 'feature', rest are cell IDs
        feature_col = data_df.columns[0]
        all_cell_cols = data_df.columns[1:]

        # Filter to only keep columns for cells that matched
        kept_cols = [feature_col] + [c for c in all_cell_cols if c in kept_cell_ids]
        filtered_data_df = data_df.select(kept_cols)

        logger.info(f"Filtered data from {len(all_cell_cols)} to {len(kept_cols) - 1} cells")

        # Determine data output path
        if in_place:
            data_output = data_path
        else:
            data_output = data_path.parent / f"{data_path.stem}_padmapped.csv"

        filtered_data_df.write_csv(data_output)
        logger.info(f"Saved filtered data to {data_output}")

    return final_output, report, data_output
