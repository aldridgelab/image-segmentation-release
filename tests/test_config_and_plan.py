from __future__ import annotations

from pathlib import Path

import pytest

from deciphaer_image_segmentation.config import load_config
from deciphaer_image_segmentation.runner import PipelineRunner
from deciphaer_image_segmentation.slurm import JobSpec, SlurmSubmitter


def test_example_config_loads() -> None:
    config = load_config("configs/example_pipeline.yaml")

    assert config.batching.n_batches == 8
    assert config.filter.enabled is True
    assert config.unet.enabled is False
    assert config.unet.crop_size == 96
    assert config.effective_crop_size == 96
    assert config.spatial_calibration.reference_pixel_microns == 0.10317
    assert config.unet.save.include_available_channels is True


def test_unet_enabled_requires_model_path(tmp_path: Path) -> None:
    filter_path = tmp_path / "filter.json"
    filter_path.write_text('{"rules": {}}', encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
run:
  output_dir: {tmp_path / "out"}
inputs:
  image_dir: {tmp_path / "images"}
filter:
  enabled: true
  config_path: {filter_path}
unet:
  enabled: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unet.enabled requires unet.model_path"):
        load_config(config_path)


def test_slurm_plan_contains_filter_crop_unet_and_final(tmp_path: Path) -> None:
    filter_path = tmp_path / "filter.json"
    filter_path.write_text('{"rules": {}}', encoding="utf-8")
    model_path = tmp_path / "best.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
run:
  output_dir: {tmp_path / "out"}
  mode: slurm
inputs:
  image_dir: {tmp_path / "images"}
batching:
  n_batches: 2
filter:
  enabled: true
  config_path: {filter_path}
unet:
  enabled: true
  model_path: {model_path}
  shards: 3
slurm:
  job_prefix: tseg
""",
        encoding="utf-8",
    )

    stages = [spec.stage for spec in PipelineRunner(load_config(config_path)).plan()]

    assert stages == ["monitor", "momia", "momia", "compile", "filter", "crop", "unet", "unet", "unet", "final"]


def test_slurm_command_includes_resources_and_dependencies(tmp_path: Path) -> None:
    spec = JobSpec(
        name="seg batch 1",
        stage="momia",
        script=tmp_path / "run_stage.sh",
        args=("repo", "config.yaml", "momia", "batch1", ""),
        dependencies=("123", "456"),
        partition="preempt",
        time="01:00:00",
        mem="8g",
        cpus_per_task=4,
    )

    command = SlurmSubmitter().command_for(spec)

    assert command[:3] == ["sbatch", "--parsable", "--job-name=seg_batch_1"]
    assert "--dependency=afterok:123:456" in command
    assert "--cpus-per-task" in command
    assert command[-5:] == ["repo", "config.yaml", "momia", "batch1", ""]
