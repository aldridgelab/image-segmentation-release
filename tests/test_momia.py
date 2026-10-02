from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from deciphaer_image_segmentation.momia import (
    _cells_from_metadata,
    _flatten_cell_row,
    _read_batch_cell_table,
    _transfer_tree,
)


def test_flatten_cell_row_extracts_bbox_centroid_and_intensity_metrics() -> None:
    row = _flatten_cell_row(
        metadata={"image_id": "img_a", "source_path": "/data/a.tif"},
        cell={
            "label": 3,
            "included": True,
            "bbox": [10, 20, 30, 40],
            "centroid": [50.5, 60.25],
            "metrics": {
                "area": 123,
                "intensity": {"Phase": 1.0, "AF405": 2.0},
                "raw_pixels": [1, 2, 3],  # list — should be ignored
            },
            "failure_reasons": ["bad_shape", "edge"],
        },
    )
    assert row["image_id"] == "img_a"
    assert row["label"] == 3
    assert row["included"] is True
    assert row["bbox_x1"] == 10 and row["bbox_y2"] == 40
    assert row["centroid_y"] == 50.5 and row["centroid_x"] == 60.25
    assert row["area"] == 123
    assert row["intensity_Phase"] == 1.0
    assert row["intensity_AF405"] == 2.0
    assert row["failure_reasons"] == "bad_shape; edge"
    assert "raw_pixels" not in row


def test_flatten_cell_row_default_cell_id_when_missing() -> None:
    row = _flatten_cell_row(
        metadata={"image_id": "img_a"},
        cell={"label": 5},
    )
    assert row["cell_id"] == "img_a_5"


def test_cells_from_metadata_reads_each_meta_json(tmp_path: Path) -> None:
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    (meta_dir / "img_a_meta.json").write_text(
        json.dumps(
            {
                "image_id": "img_a",
                "cells": [{"label": 1, "metrics": {"area": 10}}, {"label": 2}],
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "img_b_meta.json").write_text(
        json.dumps({"image_id": "img_b", "cells": [{"label": 1}]}),
        encoding="utf-8",
    )
    rows = _cells_from_metadata(meta_dir)
    assert len(rows) == 3
    assert {r["image_id"] for r in rows} == {"img_a", "img_b"}


def test_cells_from_metadata_handles_missing_dir(tmp_path: Path) -> None:
    assert _cells_from_metadata(tmp_path / "does_not_exist") == []


def test_read_batch_cell_table_prefers_cell_measurements_csv(tmp_path: Path) -> None:
    pd.DataFrame({"image_id": ["a"], "label": [1]}).to_csv(
        tmp_path / "cell_measurements.csv", index=False
    )
    df = _read_batch_cell_table(tmp_path)
    assert df is not None
    assert list(df["image_id"]) == ["a"]


def test_read_batch_cell_table_falls_back_to_metadata(tmp_path: Path) -> None:
    # Empty CSV should fall back to metadata directory.
    (tmp_path / "cell_measurements.csv").write_text("", encoding="utf-8")
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    (meta_dir / "img_a_meta.json").write_text(
        json.dumps({"image_id": "img_a", "cells": [{"label": 1}]}),
        encoding="utf-8",
    )
    df = _read_batch_cell_table(tmp_path)
    assert df is not None
    assert df.iloc[0]["image_id"] == "img_a"


def test_read_batch_cell_table_returns_none_when_nothing_available(tmp_path: Path) -> None:
    assert _read_batch_cell_table(tmp_path) is None


def test_transfer_tree_copy_mode(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("hello", encoding="utf-8")
    _transfer_tree(src, dst, mode="copy")
    assert (dst / "a.txt").is_file()
    assert (dst / "a.txt").read_text(encoding="utf-8") == "hello"


def test_transfer_tree_hardlink_mode(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("x", encoding="utf-8")
    _transfer_tree(src, dst, mode="hardlink")
    assert (dst / "a.txt").exists()
    assert os.stat(src / "a.txt").st_ino == os.stat(dst / "a.txt").st_ino


def test_transfer_tree_symlink_mode_default(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    real = src / "a.txt"
    real.write_text("x", encoding="utf-8")
    _transfer_tree(src, dst, mode="symlink")
    assert (dst / "a.txt").is_symlink()
    assert Path(os.readlink(dst / "a.txt")) == real.resolve()


def test_transfer_tree_does_not_overwrite_existing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("new", encoding="utf-8")
    (dst / "a.txt").write_text("existing", encoding="utf-8")
    _transfer_tree(src, dst, mode="copy")
    assert (dst / "a.txt").read_text(encoding="utf-8") == "existing"


def test_transfer_tree_skips_missing_src(tmp_path: Path) -> None:
    # Should not raise.
    _transfer_tree(tmp_path / "nope", tmp_path / "dst", mode="copy")
