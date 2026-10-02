"""Shared per-cell dataframe helpers used by export and extract pipelines.

These were duplicated between ``export.py`` and ``extract.py``; consolidating
prevents silent drift when one copy gets a fix and the other doesn't.
"""
from __future__ import annotations

import pandas as pd


def join_crops(cells: pd.DataFrame, crops: pd.DataFrame) -> pd.DataFrame:
    """Inner-join cell rows to U-Net crop rows on ``(image_id, label)``.

    Drops any pre-existing ``Cell_ID`` column on the cells side so the crop
    table's ``Cell_ID`` survives the merge unambiguously.
    """
    merged = cells.copy()
    if "Cell_ID" in merged.columns:
        merged = merged.drop(columns=["Cell_ID"])
    crops = crops.copy()
    crops["label"] = crops["label"].astype(int)
    merged["label"] = merged["label"].astype(int)
    return merged.merge(crops, on=["image_id", "label"], how="inner", suffixes=("", "_crop"))


def filter_unet(
    cells: pd.DataFrame,
    *,
    probability_threshold: float,
    keep_class: str,
    require_included: bool = False,
) -> pd.DataFrame:
    """Keep cells whose U-Net probability and class meet the configured cuts.

    When ``require_included`` is true, also require ``included`` (from the
    upstream filter stage) to be truthy — this is the export-pipeline
    semantics; the extract pipeline doesn't gate on ``included`` because it
    runs before that column is materialized.
    """
    if "p_cell" not in cells.columns:
        return cells.iloc[0:0].copy()
    p_cell = pd.to_numeric(cells["p_cell"], errors="coerce").fillna(0)
    mask = p_cell >= probability_threshold
    if keep_class == "cell" and "predicted_class" in cells.columns:
        mask = mask & cells["predicted_class"].astype(str).eq("cell")
    if require_included and "included" in cells.columns:
        included = cells["included"].astype(str).str.lower().isin({"true", "1", "yes"})
        mask = mask & included
    return cells.loc[mask].copy()
