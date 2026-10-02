from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from deciphaer_image_segmentation import skiplist
from deciphaer_image_segmentation.config import load_config
from deciphaer_image_segmentation.watchdog import flag_failed_batches


def _write_config(tmp_path: Path, *, n_batches: int = 2) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
run:
  name: demo
  output_dir: {tmp_path / "out"}
inputs:
  image_dir: {tmp_path / "images"}
batching:
  n_batches: {n_batches}
slurm:
  job_prefix: segv2
  monitor_enabled: false
""",
        encoding="utf-8",
    )
    return cfg


def _stage_batch(config_dir: Path, batch_name: str, image_ids: list[str]) -> None:
    """Create the input-symlink dir + masks dir for one batch."""
    batch_input = config_dir / "out" / "batches" / batch_name
    batch_input.mkdir(parents=True, exist_ok=True)
    # Make a real source file so symlinks point somewhere valid.
    for image_id in image_ids:
        real = config_dir / "images" / f"{image_id}.tif"
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_bytes(b"")
        (batch_input / f"{image_id}.tif").symlink_to(real)
    (config_dir / "out" / "momia" / "batches" / batch_name / "masks").mkdir(
        parents=True, exist_ok=True
    )


def test_flag_failed_batches_returns_empty_when_nothing_failed(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, n_batches=1)
    config = load_config(cfg_path)
    _stage_batch(tmp_path, "batch1", ["img_a"])
    # All masks present -> nothing flagged.
    mask = config.momia_batches_dir / "batch1" / "masks" / "img_a_mask.tif"
    mask.write_bytes(b"")

    skip = tmp_path / "skip.txt"
    flagged = flag_failed_batches(
        config, log_dir=None, skip_path=skip, webhook_url=None
    )
    assert flagged == []
    assert not skip.exists()


def test_flag_failed_batches_detects_missing_mask_and_appends_skiplist(
    tmp_path: Path,
) -> None:
    cfg_path = _write_config(tmp_path, n_batches=1)
    config = load_config(cfg_path)
    _stage_batch(tmp_path, "batch1", ["img_a", "img_b"])
    # Only img_a produced a mask.
    (config.momia_batches_dir / "batch1" / "masks" / "img_a_mask.tif").write_bytes(b"")

    skip = tmp_path / "skip.txt"
    # Avoid the squeue subprocess call by patching it.
    with patch(
        "deciphaer_image_segmentation.watchdog.squeue_states", return_value={}
    ):
        flagged = flag_failed_batches(
            config, log_dir=None, skip_path=skip, webhook_url=None
        )
    assert [image_id for image_id, _ in flagged] == ["img_b"]
    assert "img_b" in skiplist.load(skip)


def test_flag_failed_batches_uses_run_summary_failure_reason(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, n_batches=1)
    config = load_config(cfg_path)
    _stage_batch(tmp_path, "batch1", ["img_a"])
    summary = config.momia_batches_dir / "batch1" / "run_summary.json"
    summary.write_text(
        json.dumps(
            {"results": [{"image_id": "img_a", "status": "failed", "error": "boom: detail"}]}
        ),
        encoding="utf-8",
    )

    skip = tmp_path / "skip.txt"
    with patch(
        "deciphaer_image_segmentation.watchdog.squeue_states", return_value={}
    ):
        flagged = flag_failed_batches(
            config, log_dir=None, skip_path=skip, webhook_url=None
        )
    assert len(flagged) == 1
    image_id, reason = flagged[0]
    assert image_id == "img_a"
    assert "boom" in reason


def test_flag_failed_batches_posts_webhook_when_url_configured(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, n_batches=1)
    config = load_config(cfg_path)
    _stage_batch(tmp_path, "batch1", ["img_a"])

    skip = tmp_path / "skip.txt"
    with patch(
        "deciphaer_image_segmentation.watchdog.squeue_states", return_value={}
    ), patch(
        "deciphaer_image_segmentation.notify.post", return_value=True
    ) as post:
        flag_failed_batches(
            config,
            log_dir=None,
            skip_path=skip,
            webhook_url="https://example.invalid/webhook",
        )
    post.assert_called_once()
    url, msg = post.call_args.args
    assert url == "https://example.invalid/webhook"
    assert "img_a" in msg


def test_flag_failed_batches_skips_pending_or_running_jobs(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, n_batches=1)
    config = load_config(cfg_path)
    _stage_batch(tmp_path, "batch1", ["img_a"])
    manifest = config.output_dir / "slurm_submission_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"segv2_batch1_momia": "555"}), encoding="utf-8")

    skip = tmp_path / "skip.txt"
    with patch(
        "deciphaer_image_segmentation.watchdog.squeue_states",
        return_value={"555": "RUNNING"},
    ):
        flagged = flag_failed_batches(
            config,
            log_dir=None,
            skip_path=skip,
            webhook_url=None,
            manifest_path=manifest,
        )
    assert flagged == []
