"""
Write sidecar parquet files for vector (profiles) and event (puncta) data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


def write_profiles_parquet(
    records: list[dict[str, Any]],
    path: Path | str,
) -> None:
    """Write midline intensity profiles to a Parquet sidecar file.

    Each record should have:
    - ``cell_id``: str
    - ``channel``: str
    - ``profile_kind``: str (e.g. ``"midline_strip_mean"``)
    - ``n_points``: int
    - ``profile``: list[float]

    Args:
        records: List of profile dictionaries.
        path: Output parquet file path.
    """
    if not records:
        logger.debug("No profile records to write")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    logger.info(f"Wrote {len(df)} profile records to {path}")


def write_puncta_parquet(
    records: list[dict[str, Any]],
    path: Path | str,
) -> None:
    """Write puncta event data to a Parquet sidecar file.

    Each record should have:
    - ``cell_id``: str
    - ``channel``: str
    - ``punctum_id``: int
    - ``y_px``, ``x_px``: float (subpixel coordinates)
    - ``sigma_px``: float (spot size proxy from LoG)
    - ``peak_intensity``: float
    - ``axial_s``: float (0-1 position along midline)
    - ``radial_r``: float (distance from midline in pixels)

    Args:
        records: List of punctum dictionaries.
        path: Output parquet file path.
    """
    if not records:
        logger.debug("No puncta records to write")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    logger.info(f"Wrote {len(df)} puncta records to {path}")
