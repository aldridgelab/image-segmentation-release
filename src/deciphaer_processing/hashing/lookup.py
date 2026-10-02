"""
Cell lookup table for traceability.

Maintains a mapping from cell IDs back to source information,
stored as Parquet for efficient queries.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from deciphaer_processing.hashing.cell_hash import (
    generate_cell_hash,
    generate_cell_id,
    parse_filename_metadata,
)


class CellLookupTable:
    """
    Lookup table for cell ID -> source information mapping.

    Stores cell provenance information including:
    - cell_hash: SHA256-based unique identifier
    - cell_id: Formatted ID (e.g., "sc_a1b2c3d4...")
    - source_path: Original image path
    - mask_label: Label in the mask
    - bbox: Bounding box (min_row, min_col, max_row, max_col)
    - plate, scene, position: Parsed from filename
    - timestamp: When the cell was processed
    """

    def __init__(self, path: str | Path | None = None):
        """
        Initialize lookup table.

        Args:
            path: Optional path to existing Parquet file to load
        """
        self.path = Path(path) if path else None
        self._df: pl.DataFrame | None = None

        if self.path and self.path.exists():
            self.load(self.path)
        else:
            self._init_empty()

    def _init_empty(self) -> None:
        """Initialize an empty DataFrame with the proper schema."""
        self._df = pl.DataFrame(
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

    def add_cell(
        self,
        source_path: str | Path,
        mask_label: int,
        bbox: tuple[int, int, int, int],
        additional_fields: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a cell to the lookup table.

        Args:
            source_path: Path to source image
            mask_label: Label in the mask
            bbox: Bounding box (min_row, min_col, max_row, max_col)
            additional_fields: Optional additional metadata

        Returns:
            Generated cell_id
        """
        source_path = Path(source_path)
        cell_hash = generate_cell_hash(source_path, mask_label, bbox)
        cell_id = f"sc_{cell_hash}"

        # Parse filename metadata
        filename_meta = parse_filename_metadata(source_path.name)

        # Create row - explicitly cast to Python int to avoid numpy int64 issues
        row = {
            "cell_hash": cell_hash,
            "cell_id": cell_id,
            "source_path": str(source_path),
            "source_filename": source_path.name,
            "mask_label": int(mask_label),
            "bbox_min_row": int(bbox[0]),
            "bbox_min_col": int(bbox[1]),
            "bbox_max_row": int(bbox[2]),
            "bbox_max_col": int(bbox[3]),
            "plate": filename_meta["plate"],
            "scene": filename_meta["scene"],
            "position": filename_meta["position"],
            "timestamp": datetime.now(),
        }

        # Add additional fields if provided
        if additional_fields:
            for key, value in additional_fields.items():
                if key not in row:
                    row[key] = value

        # Append to dataframe with explicit schema to avoid type inference issues
        new_row = pl.DataFrame([row], schema={
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
        })
        self._df = pl.concat([self._df, new_row], how="diagonal")

        return cell_id

    def add_cells_batch(
        self,
        cells: list[dict[str, Any]],
    ) -> list[str]:
        """
        Add multiple cells efficiently.

        Args:
            cells: List of dicts with keys:
                - source_path: Path to source image
                - mask_label: Label in the mask
                - bbox: Bounding box tuple

        Returns:
            List of generated cell_ids
        """
        rows = []
        cell_ids = []

        for cell in cells:
            source_path = Path(cell["source_path"])
            mask_label = cell["mask_label"]
            bbox = cell["bbox"]

            cell_hash = generate_cell_hash(source_path, mask_label, bbox)
            cell_id = f"sc_{cell_hash}"
            cell_ids.append(cell_id)

            # Parse filename metadata
            filename_meta = parse_filename_metadata(source_path.name)

            rows.append({
                "cell_hash": cell_hash,
                "cell_id": cell_id,
                "source_path": str(source_path),
                "source_filename": source_path.name,
                "mask_label": int(mask_label),
                "bbox_min_row": int(bbox[0]),
                "bbox_min_col": int(bbox[1]),
                "bbox_max_row": int(bbox[2]),
                "bbox_max_col": int(bbox[3]),
                "plate": filename_meta["plate"],
                "scene": filename_meta["scene"],
                "position": filename_meta["position"],
                "timestamp": datetime.now(),
            })

        if rows:
            new_df = pl.DataFrame(rows)
            self._df = pl.concat([self._df, new_df], how="diagonal")

        return cell_ids

    def dehash_cell(self, cell_id: str) -> dict[str, Any] | None:
        """
        Look up source information for a cell ID.

        Args:
            cell_id: Cell ID string (e.g., "sc_a1b2c3d4...")

        Returns:
            Dictionary with source information, or None if not found
        """
        # Handle both "sc_hash" and raw "hash" formats
        if cell_id.startswith("sc_"):
            cell_hash = cell_id[3:]
        else:
            cell_hash = cell_id

        result = self._df.filter(
            (pl.col("cell_id") == cell_id) | (pl.col("cell_hash") == cell_hash)
        )

        if len(result) == 0:
            return None

        row = result.row(0, named=True)
        return {
            "cell_id": row["cell_id"],
            "cell_hash": row["cell_hash"],
            "source_path": row["source_path"],
            "source_filename": row["source_filename"],
            "mask_label": row["mask_label"],
            "bbox": (
                row["bbox_min_row"],
                row["bbox_min_col"],
                row["bbox_max_row"],
                row["bbox_max_col"],
            ),
            "plate": row["plate"],
            "scene": row["scene"],
            "position": row["position"],
            "timestamp": row["timestamp"],
        }

    def get_cells_by_plate(self, plate: str) -> pl.DataFrame:
        """Get all cells from a specific plate."""
        return self._df.filter(pl.col("plate") == plate)

    def get_cells_by_scene(self, plate: str, scene: str) -> pl.DataFrame:
        """Get all cells from a specific plate and scene."""
        return self._df.filter(
            (pl.col("plate") == plate) & (pl.col("scene") == scene)
        )

    def save(self, path: str | Path | None = None) -> None:
        """
        Save lookup table to Parquet file.

        Args:
            path: Output path (uses self.path if not specified)
        """
        save_path = Path(path) if path else self.path
        if save_path is None:
            raise ValueError("No path specified for saving")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._df.write_parquet(save_path)
        logger.info(f"Saved lookup table with {len(self._df)} cells to {save_path}")

    def load(self, path: str | Path) -> None:
        """
        Load lookup table from Parquet file.

        Args:
            path: Path to Parquet file
        """
        self._df = pl.read_parquet(path)
        self.path = Path(path)
        logger.info(f"Loaded lookup table with {len(self._df)} cells from {path}")

    @property
    def df(self) -> pl.DataFrame:
        """Get the underlying DataFrame."""
        return self._df

    def __len__(self) -> int:
        """Return number of cells in the table."""
        return len(self._df)

    def __contains__(self, cell_id: str) -> bool:
        """Check if a cell ID exists in the table."""
        return self.dehash_cell(cell_id) is not None
