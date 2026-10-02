from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from deciphaer_image_segmentation.monitor import (
    _emit_stall_warning,
    _load_state,
    _save_state,
)


def test_load_state_returns_empty_dict_when_missing(tmp_path: Path) -> None:
    assert _load_state(tmp_path / "state.json") == {}


def test_load_state_recovers_from_corrupt_json(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text("{ not valid json", encoding="utf-8")
    assert _load_state(p) == {}


def test_load_state_returns_empty_when_top_level_not_dict(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert _load_state(p) == {}


def test_save_state_roundtrips(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    payload = {"batch1": {"last_count": 5, "warned": False}}
    _save_state(p, payload)
    assert json.loads(p.read_text(encoding="utf-8")) == payload


def test_emit_stall_warning_posts_to_webhook_when_configured() -> None:
    with patch("deciphaer_image_segmentation.notify.post", return_value=True) as post:
        _emit_stall_warning(
            webhook_url="https://example.invalid/webhook",
            run_name="demo",
            batch_name="batch7",
            job_id="999",
            mask_count=3,
            stalled_for_minutes=6.5,
        )
    post.assert_called_once()
    url, msg = post.call_args.args
    assert url == "https://example.invalid/webhook"
    assert "batch7" in msg
    assert "999" in msg
    assert "3 mask" in msg


def test_emit_stall_warning_skips_post_when_no_webhook() -> None:
    with patch("deciphaer_image_segmentation.notify.post") as post:
        _emit_stall_warning(
            webhook_url=None,
            run_name="demo",
            batch_name="b",
            job_id="1",
            mask_count=0,
            stalled_for_minutes=10.0,
        )
    post.assert_not_called()
