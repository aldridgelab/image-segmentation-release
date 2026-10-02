from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from deciphaer_image_segmentation.crops import crop_cells_from_table
from deciphaer_image_segmentation.export import finalize_outputs
from deciphaer_image_segmentation.filtering import run_filter_stage


def test_filter_stage_applies_rod_rules(tmp_path: Path) -> None:
    table = tmp_path / "cells.csv"
    pd.DataFrame(
        [
            {"image_id": "a_Phase", "label": 1, "included": True, "area_px": 150, "aspect_ratio": 3.0, "touching_edge": False},
            {"image_id": "a_Phase", "label": 2, "included": True, "area_px": 20, "aspect_ratio": 3.0, "touching_edge": False},
            {"image_id": "a_Phase", "label": 3, "included": False, "area_px": 150, "aspect_ratio": 3.0, "touching_edge": False},
        ]
    ).to_csv(table, index=False)
    filter_config = tmp_path / "filter.json"
    filter_config.write_text(
        json.dumps({"rules": {"area_px": {"min": 100, "max": 500}, "aspect_ratio": {"min": 2}, "touching_edge": False}}),
        encoding="utf-8",
    )

    result = run_filter_stage(
        cell_table_path=table,
        filter_config_path=filter_config,
        output_dir=tmp_path / "filter",
        require_momia_included=True,
    )

    filtered = pd.read_csv(result.table_path)
    assert result.n_passed == 1
    assert filtered["included"].tolist() == [True, False, False]
    assert "area_px:below_min" in filtered.loc[1, "filter_reasons"]
    assert "momia_excluded" in filtered.loc[2, "filter_reasons"]


def test_crop_stage_writes_hashed_fixed_size_stack(tmp_path: Path) -> None:
    source = tmp_path / "sample_Phase.tif"
    image = np.arange(100, dtype=np.uint16).reshape(10, 10)
    tifffile.imwrite(source, image)
    tifffile.imwrite(tmp_path / "sample_HADA.tif", image + 1000)
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    mask = np.zeros((10, 10), dtype=np.uint16)
    mask[1:3, 1:3] = 7
    tifffile.imwrite(masks_dir / "sample_Phase_mask.tif", mask)
    table = tmp_path / "cells.csv"
    pd.DataFrame(
        [
            {
                "image_id": "sample_Phase",
                "source_path": str(source),
                "label": 7,
                "cell_id": "sample_Phase_7",
                "included": True,
                "bbox_x1": 1,
                "bbox_y1": 1,
                "bbox_x2": 3,
                "bbox_y2": 3,
                "centroid_y": 2,
                "centroid_x": 2,
            }
        ]
    ).to_csv(table, index=False)

    manifest = crop_cells_from_table(
        cell_table_path=table,
        masks_dir=masks_dir,
        crops_dir=tmp_path / "crops",
        selected_channels=("Phase", "Mask"),
        phase_channel="Phase",
        crop_size=6,
        min_bbox_pad=1,
        edge_pad_mode="reflect",
        workers=1,
        include_available_channels=True,
        segmentation_run_hash="runabc123",
    )

    rows = pd.read_csv(manifest)
    assert len(rows) == 1
    assert rows.loc[0, "Cell_ID"].startswith("sc_")
    assert rows.loc[0, "cell_hash"] == f"sha256:{rows.loc[0, 'Cell_ID'].removeprefix('sc_')}"
    assert rows.loc[0, "segmentation_run_hash"] == "runabc123"
    assert rows.loc[0, "source_image"] == "sample"
    assert rows.loc[0, "channels"] == "Phase;HADA;Mask"
    stack = tifffile.imread(rows.loc[0, "crop_path"])
    assert stack.shape == (3, 6, 6)
    assert stack[1, 3, 3] == image[2, 2] + 1000
    assert stack[-1].max() == 7
    with tifffile.TiffFile(rows.loc[0, "crop_path"]) as tif:
        assert tif.is_ome
        assert tif.series[0].axes == "CYX"
        metadata = _parse_ome_metadata(tif.ome_metadata or "")
    assert metadata["channel_names"] == ["Phase", "HADA", "Mask"]
    assert metadata["map_values"]["deciphaer.channel_labels"] == "Phase;HADA;Mask"
    info = json.loads(metadata["map_values"]["deciphaer.provenance_json"])
    assert info["cell_hash"] == rows.loc[0, "cell_hash"]
    assert info["segmentation_run_hash"] == "runabc123"
    assert [channel["label"] for channel in info["channels"]] == ["Phase", "HADA", "Mask"]


def test_finalize_outputs_writes_scmorph_files_with_unet_filter(tmp_path: Path) -> None:
    cell_table = tmp_path / "cells.csv"
    pd.DataFrame(
        [
            {"image_id": "sample_Phase", "source_path": "sample_Phase.tif", "label": 1, "included": True, "area_px": 123, "aspect_ratio": 3.1},
            {"image_id": "sample_Phase", "source_path": "sample_Phase.tif", "label": 2, "included": True, "area_px": 456, "aspect_ratio": 1.2},
        ]
    ).to_csv(cell_table, index=False)
    crop_manifest = tmp_path / "crops_manifest.csv"
    pd.DataFrame(
        [
            {
                "Cell_ID": "sc_pass",
                "cell_hash": "sha256:pass",
                "segmentation_run_hash": "runabc123",
                "source_image": "sample",
                "image_id": "sample_Phase",
                "label": 1,
                "crop_path": "sc_pass.tif",
                "bbox_min_row": 1,
                "bbox_min_col": 2,
                "bbox_max_row": 3,
                "bbox_max_col": 4,
                "channels": "Phase;HADA;Mask",
            },
            {
                "Cell_ID": "sc_fail",
                "cell_hash": "sha256:fail",
                "segmentation_run_hash": "runabc123",
                "source_image": "sample",
                "image_id": "sample_Phase",
                "label": 2,
                "crop_path": "sc_fail.tif",
                "bbox_min_row": 5,
                "bbox_min_col": 6,
                "bbox_max_row": 7,
                "bbox_max_col": 8,
            },
        ]
    ).to_csv(crop_manifest, index=False)
    unet = tmp_path / "unet.csv"
    pd.DataFrame(
        [
            {"Cell_ID": "sc_pass", "predicted_class": "cell", "p_cell": 0.9, "p_background": 0.05, "p_noncell": 0.05},
            {"Cell_ID": "sc_fail", "predicted_class": "non_cell", "p_cell": 0.1, "p_background": 0.1, "p_noncell": 0.8},
        ]
    ).to_csv(unet, index=False)

    outputs = finalize_outputs(
        cell_table_path=cell_table,
        crop_manifest_path=crop_manifest,
        output_dir=tmp_path / "final",
        prefix="sc_morph",
        feature_columns=(),
        unet_classifications_path=unet,
        probability_threshold=0.5,
        keep_class="cell",
        segmentation_run_hash="runabc123",
    )

    final = pd.read_csv(outputs["cell_measurements_final"])
    data = pd.read_csv(outputs["data"])
    meta = pd.read_csv(outputs["meta"])
    mapping = pd.read_csv(outputs["mapping"])
    assert final["Cell_ID"].tolist() == ["sc_pass"]
    assert "area_px" in data["feature"].tolist()
    assert meta["Cell_ID"].tolist() == ["sc_pass"]
    assert meta.loc[0, "cell_hash"] == "sha256:pass"
    assert meta.loc[0, "segmentation_run_hash"] == "runabc123"
    assert meta.loc[0, "source_image"] == "sample"
    assert mapping.loc[0, "mask_label"] == 1
    assert mapping.loc[0, "channels"] == "Phase;HADA;Mask"


def _parse_ome_metadata(xml: str) -> dict[str, object]:
    root = ET.fromstring(xml)
    ns = {"ome": root.tag[1:].split("}")[0]}
    channels = root.findall(".//ome:Channel", ns)
    map_entries = root.findall(".//ome:MapAnnotation/ome:Value/ome:M", ns)
    return {
        "channel_names": [channel.attrib.get("Name", "") for channel in channels],
        "map_values": {entry.attrib["K"]: entry.text or "" for entry in map_entries},
    }
