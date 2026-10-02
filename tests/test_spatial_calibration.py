from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from deciphaer_image_segmentation.config import load_config
from deciphaer_image_segmentation.filtering import resolve_filter_rules, run_filter_stage
from deciphaer_image_segmentation.momia import effective_momia_settings, run_momia_batch
from deciphaer_image_segmentation.runner import PipelineRunner
from deciphaer_image_segmentation.spatial import scale_momia_settings, scaled_integer
from momia_seg.batch import DEFAULT_SETTINGS, validate_settings


def make_config(tmp_path: Path, **sections):
    raw = {
        "inputs": {"image_dir": str(tmp_path / "images"), "pixel_microns": 0.0645},
        "run": {"output_dir": str(tmp_path / "out"), "keep_intermediate": True},
        **sections,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def test_legacy_settings_remain_in_input_pixels(tmp_path):
    config = make_config(tmp_path)
    assert config.spatial_scale == 1
    assert config.effective_crop_size == 96
    assert config.effective_min_bbox_pad == 4
    assert config.crops_dir.name == "cell_crops_96"
    expected = copy.deepcopy(DEFAULT_SETTINGS)
    expected["image"]["pixel_microns"] = 0.0645
    assert effective_momia_settings(config) == expected


def test_reference_calibration_resolves_crop_and_momia_settings(tmp_path):
    config = make_config(tmp_path, spatial_calibration={"reference_pixel_microns": 0.10317})
    factor = 0.10317 / 0.0645
    assert config.spatial_scale == pytest.approx(factor)
    assert config.unet.crop_size == 96  # requested value is not mutated
    assert config.effective_crop_size == 154
    assert config.effective_min_bbox_pad == 6
    assert config.crops_dir.name == "cell_crops_154"
    settings = effective_momia_settings(config)
    assert settings["segmentation"]["min_particle_size"] == 128
    assert settings["segmentation"]["window_size"] == 15
    assert settings["filtering"]["edge_distance"] == 13
    assert settings["filtering"]["area_max"] == pytest.approx(500 * factor**2)
    # The user's rounded reference is distinct from the repository reference.
    rounded = replace(config, spatial_calibration=replace(config.spatial_calibration, reference_pixel_microns=0.103))
    assert rounded.effective_crop_size == 153


@pytest.mark.parametrize("factor", [0.25, 0.5, 1, 1.6, 2])
def test_momia_scaling_is_selective_valid_and_nonmutating(factor):
    source = copy.deepcopy(DEFAULT_SETTINGS)
    source["filtering"].update(length_min=1.0, width_max=1.2, area_max=None)
    source["segmentation"]["min_hole_size"] = 0
    original = copy.deepcopy(source)
    result = scale_momia_settings(source, factor)
    assert source == original
    assert result["image"]["max_drift"] == 8 * factor
    assert result["centrality"]["center_window_px"] == scaled_integer(20, factor)
    assert result["filtering"]["area_min"] == pytest.approx(100 * factor**2)
    assert result["filtering"]["area_max"] is None
    assert result["segmentation"]["min_hole_size"] == 0
    assert result["filtering"]["length_min"] == 1.0
    assert result["filtering"]["width_max"] == 1.2
    assert result["filtering"]["solidity_max"] == 1.0
    assert result["segmentation"]["k"] == 0.05
    assert result["outputs"] == source["outputs"]
    validate_settings(result)
    if factor == 1:
        assert result == original


def test_momia_scaling_accepts_numeric_strings_in_float_settings():
    source = copy.deepcopy(DEFAULT_SETTINGS)
    source["image"]["max_drift"] = "8"
    source["filtering"]["area_max"] = "500"
    result = scale_momia_settings(source, 2)
    assert result["image"]["max_drift"] == 16
    assert result["filtering"]["area_max"] == 2000


def test_integer_rounding_and_minimum_windows():
    assert scaled_integer(5, 0.5) == 3
    assert scaled_integer(0, 2) == 0
    assert scaled_integer(1, 0.01, minimum=1) == 1
    assert scaled_integer(9, 0.01, minimum=3, odd=True) == 3
    assert scaled_integer(9, 2, minimum=3, odd=True) == 19  # odd tie rounds up
    assert scaled_integer(9, 1.6, minimum=3, odd=True) == 15


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("section,key", [("inputs", "pixel_microns"), ("spatial_calibration", "reference_pixel_microns")])
def test_invalid_calibration_is_rejected(tmp_path, value, section, key):
    values = {key: value}
    if section == "inputs":
        values["image_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match="finite and > 0"):
        make_config(tmp_path, **{section: values})


def test_unknown_calibration_key_does_not_silently_disable_scaling(tmp_path):
    with pytest.raises(ValueError, match="Unknown spatial_calibration"):
        make_config(tmp_path, spatial_calibration={"reference_pixel_micron": 0.1})


@pytest.mark.parametrize("unet", [{"crop_size": 0}, {"min_bbox_pad": -1}])
def test_invalid_crop_geometry_is_rejected_even_when_disabled(tmp_path, unet):
    with pytest.raises(ValueError, match="unet\\."):
        make_config(tmp_path, unet=unet)


@pytest.mark.parametrize("payload", [
    {"rules": {"area_px": {"min": 60, "max": 300}, "length_um": {"min": 1}}},
    {"area_min": 60, "area_max": 300, "length_min": 1},
    {"rules": {"area": {"min": 60, "max": 300}, "length": {"min": 1}}},
])
def test_filter_scales_only_known_pixel_area_bounds(payload):
    before = copy.deepcopy(payload)
    rules = resolve_filter_rules(payload, spatial_scale=2)
    assert rules == {"area_px": {"min": 240, "max": 1200}, "length_um": {"min": 1}}
    assert payload == before


def test_filter_preserves_physical_dimensionless_and_custom_rules():
    rules = {"area_um2": {"min": 1}, "width_um": {"max": 2}, "solidity": {"min": 0.7}, "custom_px": {"min": 4}}
    assert resolve_filter_rules(rules, spatial_scale=2) == rules
    assert resolve_filter_rules({"area_px": {"min": None}}, spatial_scale=2) == {"area_px": {"min": None}}
    with pytest.raises(ValueError, match="min/max"):
        resolve_filter_rules({"area_px": 10}, spatial_scale=2)


def test_filter_decisions_preserve_physical_area_and_record_effective_rules(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": {"area_px": {"min": 60, "max": 300}}}))
    for factor in (1, 2):
        table = tmp_path / f"cells{factor}.csv"
        pd.DataFrame({"area_px": np.array([30, 100, 400]) * factor**2}).to_csv(table, index=False)
        result = run_filter_stage(
            cell_table_path=table, filter_config_path=rules_path, output_dir=tmp_path / f"filter{factor}",
            require_momia_included=False, spatial_scale=factor,
        )
        assert pd.read_csv(result.table_path)["included"].tolist() == [False, True, False]
        summary = json.loads(result.summary_path.read_text())
        assert summary["effective_rules"]["area_px"]["min"] == 60 * factor**2


def test_momia_batch_uses_and_records_effective_settings(tmp_path, monkeypatch):
    from deciphaer_image_segmentation import momia
    from momia_seg.batch import BatchRunSummary

    config = make_config(tmp_path, spatial_calibration={"reference_pixel_microns": 0.10317})
    captured = {}

    def fake_run_batch(*, settings, settings_path):
        captured.update(settings)
        assert json.loads(settings_path.read_text()) == settings
        return BatchRunSummary("", "", "", "", "", False, 0, None, 0, 0, 0, 0, True, str(settings_path), [])

    monkeypatch.setattr(momia, "run_batch", fake_run_batch)
    run_momia_batch(config, "batch1")
    assert captured["segmentation"]["min_particle_size"] == 128
    assert captured["image"]["pixel_microns"] == 0.0645


@pytest.mark.parametrize("mode", ["local", "slurm"])
@pytest.mark.parametrize("shard", [None, "0/1"])
def test_runner_native_crops_preserve_physical_coverage_and_mask_labels(tmp_path, mode, shard):
    for factor in (1, 2):
        work = tmp_path / str(factor)
        work.mkdir()
        config = make_config(
            work,
            inputs={"image_dir": str(work), "pixel_microns": 0.1 / factor},
            spatial_calibration={"reference_pixel_microns": 0.1},
            run={"output_dir": str(work / "out"), "mode": mode},
        )
        size = 128 * factor
        image = np.arange(size * size, dtype=np.uint16).reshape(size, size)
        source = work / "scene_Phase.tif"
        tifffile.imwrite(source, image)
        masks = config.momia_compiled_dir / "masks"
        masks.mkdir(parents=True)
        mask = np.zeros_like(image)
        y0, x0, y1, x1 = np.array([60, 56, 66, 72]) * factor
        mask[y0:y1, x0:x1] = 7
        tifffile.imwrite(masks / "scene_Phase_mask.tif", mask)
        pd.DataFrame([{
            "image_id": "scene_Phase", "source_path": str(source), "label": 7, "included": True,
            "bbox_x1": x0, "bbox_y1": y0, "bbox_x2": x1, "bbox_y2": y1,
        }]).to_csv(config.momia_compiled_dir / "cell_measurements.csv", index=False)
        runner = PipelineRunner(config)
        manifest = pd.read_csv(runner.run_crop(shard=shard))
        crop = tifffile.imread(manifest.loc[0, "crop_path"])
        assert crop.shape == (2, 96 * factor, 96 * factor)
        assert set(np.unique(crop[-1])) == {0, 7}
        assert (crop[-1] == 7).sum() == 96 * factor**2
        assert crop.shape[-1] * config.inputs.pixel_microns == pytest.approx(9.6)
        settings_name = "crop_settings.json" if shard is None else "crop_settings_shard_0_of_1.json"
        crop_settings = json.loads((config.unet_dir / settings_name).read_text())
        assert crop_settings["crop_size_um"] == pytest.approx(9.6)
        assert crop_settings["crop_size_px"] == 96 * factor
        assert runner._config_snapshot()["resolved_spatial_settings"]["unet"] == crop_settings


def test_snapshot_contains_requested_and_effective_values_and_changes_hash(tmp_path):
    filter_path = tmp_path / "filter.json"
    filter_path.write_text('{"rules": {"area_px": {"min": 60}}}')
    config = make_config(tmp_path, filter={"enabled": True, "config_path": str(filter_path)},
                         spatial_calibration={"reference_pixel_microns": 0.10317})
    runner = PipelineRunner(config)
    snapshot = runner._config_snapshot()
    assert snapshot["unet"]["crop_size"] == 96
    assert snapshot["resolved_spatial_settings"]["unet"]["crop_size_px"] == 154
    assert snapshot["resolved_spatial_settings"]["filter_rules"]["area_px"]["min"] == pytest.approx(60 * config.spatial_scale**2)
    legacy = replace(config, spatial_calibration=replace(config.spatial_calibration, reference_pixel_microns=None))
    assert runner.segmentation_run_hash() != PipelineRunner(legacy).segmentation_run_hash()
