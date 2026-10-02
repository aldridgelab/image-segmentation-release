from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deciphaer_image_segmentation.cell_utils import filter_unet, join_crops

_log = logging.getLogger(__name__)


def finalize_outputs(
    *,
    cell_table_path: Path,
    crop_manifest_path: Path | None,
    output_dir: Path,
    prefix: str,
    feature_columns: tuple[str, ...],
    unet_classifications_path: Path | None = None,
    probability_threshold: float = 0.5,
    keep_class: str = "cell",
    include_unet_probabilities: bool = True,
    padmap_path: Path | None = None,
    padmap_id_pattern: str | None = None,
    padmap_id_source: str = "source_path",
    padmap_join: str = "left",
    passed_crops_dir: Path | None = None,
    segmentation_run_hash: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(cell_table_path)
    n_input_cells = len(cells)
    _log.info("final: loaded %d cells from %s", n_input_cells, cell_table_path)
    if crop_manifest_path and crop_manifest_path.exists():
        crops = pd.read_csv(crop_manifest_path)
        cells = join_crops(cells, crops)
        _log.info("final: joined crops manifest (%d crops) → %d rows after join", len(crops), len(cells))
    else:
        cells = _ensure_cell_ids(cells)
    cells = _ensure_cell_provenance(cells, segmentation_run_hash=segmentation_run_hash)
    if unet_classifications_path is not None:
        unet = pd.read_csv(unet_classifications_path)
        before = len(cells)
        cells = cells.merge(unet, on="Cell_ID", how="left", suffixes=("", "_unet"))
        cells = filter_unet(
            cells,
            probability_threshold=probability_threshold,
            keep_class=keep_class,
            require_included=True,
        )
        _log.info(
            "final: U-Net merged (%d preds), kept %d/%d cells (threshold=%.2f, class=%s)",
            len(unet),
            len(cells),
            before,
            probability_threshold,
            keep_class,
        )
    else:
        if "included" in cells.columns:
            included = cells["included"].astype(str).str.lower().isin({"true", "1", "yes"})
            cells = cells.loc[included].copy()

    if padmap_path is not None:
        before = len(cells)
        cells = _apply_padmap(
            cells,
            padmap_path=padmap_path,
            pattern=padmap_id_pattern or "",
            source_col=padmap_id_source,
            join=padmap_join,
        )
        n_unk = int((cells["drug"] == "UNK").sum()) if "drug" in cells.columns else 0
        _log.info(
            "final: padmap applied (%s, %d/%d matched, %d UNK)",
            padmap_path,
            before - n_unk,
            before,
            n_unk,
        )
    else:
        cells["drug"] = cells.get("drug", "UNK")
        cells["GroupLabels_Strict"] = cells.get("GroupLabels_Strict", "UNK")

    final_table = output_dir / "cell_measurements_final.csv"
    cells.to_csv(final_table, index=False)
    if passed_crops_dir is not None and "crop_path" in cells.columns:
        _copy_passed_crops(cells, passed_crops_dir)
        _log.info("final: linked %d passed crops → %s", len(cells), passed_crops_dir)

    features = _build_feature_matrix(
        cells,
        feature_columns=feature_columns,
        include_unet_probabilities=include_unet_probabilities,
    )
    data_path = output_dir / f"{prefix}_data.csv"
    features.to_csv(data_path, index=False)

    meta_path = output_dir / f"{prefix}_meta.csv"
    cells[_meta_columns(cells)].to_csv(meta_path, index=False)

    mapping_path = output_dir / f"{prefix}_mapping.csv"
    _mapping_table(cells).to_csv(mapping_path, index=False)

    summary_path = output_dir / "final_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "n_cells": int(len(cells)),
                "segmentation_run_hash": segmentation_run_hash or "",
                "outputs": {
                    "cell_measurements_final": str(final_table),
                    "data": str(data_path),
                    "meta": str(meta_path),
                    "mapping": str(mapping_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log.info(
        "final: wrote %d cells — data=%s, meta=%s, mapping=%s",
        len(cells),
        data_path,
        meta_path,
        mapping_path,
    )
    return {
        "cell_measurements_final": final_table,
        "data": data_path,
        "meta": meta_path,
        "mapping": mapping_path,
        "summary": summary_path,
    }


def _ensure_cell_ids(cells: pd.DataFrame) -> pd.DataFrame:
    out = cells.copy()
    if "Cell_ID" in out.columns:
        return out
    out["Cell_ID"] = [_hash_row(row, idx) for idx, row in out.iterrows()]
    return out


def _ensure_cell_provenance(cells: pd.DataFrame, *, segmentation_run_hash: str | None) -> pd.DataFrame:
    out = cells.copy()
    if "Cell_ID" in out.columns:
        fallback_hash = "sha256:" + out["Cell_ID"].astype(str).str.replace(r"^sc_", "", regex=True)
        if "cell_hash" not in out.columns:
            out["cell_hash"] = fallback_hash
        else:
            missing_hash = out["cell_hash"].isna() | out["cell_hash"].astype(str).eq("")
            out.loc[missing_hash, "cell_hash"] = fallback_hash.loc[missing_hash]
    if "segmentation_run_hash" not in out.columns:
        out["segmentation_run_hash"] = segmentation_run_hash or ""
    elif segmentation_run_hash:
        missing = out["segmentation_run_hash"].isna() | out["segmentation_run_hash"].astype(str).eq("")
        out.loc[missing, "segmentation_run_hash"] = segmentation_run_hash
    fallback_source_image = pd.Series(
        [_source_image(row) for _, row in out.iterrows()],
        index=out.index,
    )
    if "source_image" not in out.columns:
        out["source_image"] = fallback_source_image
    else:
        missing_source = out["source_image"].isna() | out["source_image"].astype(str).eq("")
        out.loc[missing_source, "source_image"] = fallback_source_image.loc[missing_source]
    return out


def _hash_row(row: pd.Series, idx: int) -> str:
    source = Path(str(row.get("source_path", ""))).name
    label = int(row.get("label", 0) or 0)
    bbox = (
        row.get("bbox_y1", row.get("bbox_min_row", "")),
        row.get("bbox_x1", row.get("bbox_min_col", "")),
        row.get("bbox_y2", row.get("bbox_max_row", "")),
        row.get("bbox_x2", row.get("bbox_max_col", "")),
    )
    raw = f"{source}|{label}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}|{idx}"
    return f"sc_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _apply_padmap(
    cells: pd.DataFrame,
    *,
    padmap_path: Path,
    pattern: str,
    source_col: str,
    join: str,
) -> pd.DataFrame:
    padmap = pd.read_csv(padmap_path)
    if {"drug", "id"} - set(padmap.columns):
        raise ValueError(f"Padmap must contain columns 'drug' and 'id': {padmap_path}")
    rx = re.compile(pattern)
    out = cells.copy()
    if source_col not in out.columns:
        raise KeyError(f"Padmap source column {source_col!r} missing from final table")
    out["padmap_id"] = out[source_col].astype(str).map(lambda value: _regex_first(rx, value))
    out = out.merge(padmap[["drug", "id"]].rename(columns={"id": "padmap_id"}), on="padmap_id", how=join)
    if join == "left":
        out["drug"] = out["drug"].fillna("UNK")
    out["GroupLabels_Strict"] = out["drug"]
    return out


def _build_feature_matrix(
    cells: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    include_unet_probabilities: bool,
) -> pd.DataFrame:
    if feature_columns:
        features = [column for column in feature_columns if column in cells.columns]
    else:
        blocked = {
            "label",
            "batch",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "bbox_min_row",
            "bbox_min_col",
            "bbox_max_row",
            "bbox_max_col",
            "crop_size_y",
            "crop_size_x",
        }
        features = [
            column
            for column in cells.columns
            if column not in blocked and _is_numeric_series(cells[column])
        ]
        if not include_unet_probabilities:
            features = [
                column
                for column in features
                if column not in {"p_background", "p_cell", "p_noncell", "confidence"}
            ]
    data: dict[str, Any] = {"feature": sorted(features)}
    for _, row in cells.iterrows():
        data[str(row["Cell_ID"])] = [row.get(feature, np.nan) for feature in data["feature"]]
    return pd.DataFrame(data)


def _mapping_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cells.iterrows():
        rows.append(
            {
                "Cell_ID": row.get("Cell_ID", ""),
                "cell_hash": row.get("cell_hash", ""),
                "segmentation_run_hash": row.get("segmentation_run_hash", ""),
                "original_cell_id": row.get("original_cell_id", row.get("cell_id", "")),
                "source_image": row.get("source_image", ""),
                "source_path": row.get("source_path", ""),
                "source_filename": Path(str(row.get("source_path", ""))).name,
                "mask_label": row.get("label", ""),
                "image_id": row.get("image_id", ""),
                "drug": row.get("drug", "UNK"),
                "GroupLabels_Strict": row.get("GroupLabels_Strict", "UNK"),
                "bbox_min_row": row.get("bbox_min_row", row.get("bbox_y1", "")),
                "bbox_min_col": row.get("bbox_min_col", row.get("bbox_x1", "")),
                "bbox_max_row": row.get("bbox_max_row", row.get("bbox_y2", "")),
                "bbox_max_col": row.get("bbox_max_col", row.get("bbox_x2", "")),
                "crop_path": row.get("crop_path", ""),
                "channels": row.get("channels", ""),
                "channel_metadata": row.get("channel_metadata", ""),
            }
        )
    return pd.DataFrame(rows)


def _meta_columns(cells: pd.DataFrame) -> list[str]:
    preferred = [
        "Cell_ID",
        "cell_hash",
        "segmentation_run_hash",
        "source_image",
        "source_path",
        "image_id",
        "label",
        "crop_path",
        "drug",
        "GroupLabels_Strict",
    ]
    return [column for column in preferred if column in cells.columns]


def _source_image(row: pd.Series) -> str:
    source = Path(str(row.get("source_path", ""))).stem
    image_id = str(row.get("image_id", ""))
    for value in (image_id, source):
        if value.endswith("_Phase"):
            return value[: -len("_Phase")]
        if value:
            return value
    return ""


def _copy_passed_crops(cells: pd.DataFrame, dst_dir: Path) -> None:
    """Hardlink passed crops into final/passed_crops; fall back to copy across filesystems.

    Hardlinks let us delete the source crops_dir during intermediate cleanup without
    losing the final crops, and they cost roughly nothing on the same filesystem.
    """
    import os

    dst_dir.mkdir(parents=True, exist_ok=True)
    for crop_path in cells["crop_path"].dropna().astype(str):
        src = Path(crop_path)
        if not src.exists():
            continue
        dst = dst_dir / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _regex_first(rx: re.Pattern[str], value: str) -> str | None:
    match = rx.search(value)
    return match.group(1) if match else None


def _is_numeric_series(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().any())
