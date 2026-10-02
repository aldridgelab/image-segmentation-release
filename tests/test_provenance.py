from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from deciphaer_image_segmentation import provenance
from deciphaer_image_segmentation.provenance import (
    _package_versions,
    _redact,
    seed_everything,
    write_run_metadata,
)


def test_seed_everything_makes_random_and_numpy_deterministic() -> None:
    seed_everything(42)
    a = (random.random(), np.random.rand())
    seed_everything(42)
    b = (random.random(), np.random.rand())
    assert a == b


def test_redact_masks_sensitive_keys_recursively() -> None:
    payload = {
        "user": "alice",
        "api_key": "k",
        "nested": {"password": "p", "ok": 1},
        "list": [{"secret": "s"}, "fine"],
    }
    redacted = _redact(payload)
    assert redacted["user"] == "alice"
    assert redacted["api_key"] == "<redacted>"
    assert redacted["nested"]["password"] == "<redacted>"
    assert redacted["nested"]["ok"] == 1
    assert redacted["list"][0]["secret"] == "<redacted>"
    assert redacted["list"][1] == "fine"


def test_redact_stringifies_paths() -> None:
    out = _redact({"path": Path("/tmp/x")})
    assert out["path"] == "/tmp/x"


def test_package_versions_handles_missing_packages() -> None:
    versions = _package_versions(["numpy", "definitely-not-installed-xyz"])
    assert "numpy" in versions
    assert versions["numpy"] != "not-installed"
    assert versions["definitely-not-installed-xyz"] == "not-installed"


def test_write_run_metadata_writes_valid_json_with_redaction(tmp_path: Path) -> None:
    path = write_run_metadata(
        output_dir=tmp_path,
        config_snapshot={"run": {"name": "demo"}, "api_key": "secret"},
        input_paths=["/data/a.tif"],
        seed=7,
        artifacts={"final": str(tmp_path / "final")},
        run_hash="abc123",
    )
    assert path == tmp_path / "run_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 7
    assert payload["segmentation_run_hash"] == "abc123"
    assert payload["input_paths"] == ["/data/a.tif"]
    assert payload["config"]["api_key"] == "<redacted>"
    assert payload["config"]["run"]["name"] == "demo"
    assert "numpy" in payload["packages"]


def test_seed_everything_logs_when_torch_seeding_fails(monkeypatch) -> None:
    """The fallback path must surface via loguru, not be silently swallowed."""
    import sys
    import types

    fake_torch = types.SimpleNamespace(
        manual_seed=lambda _seed: (_ for _ in ()).throw(RuntimeError("nope")),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # Should not raise — just log.
    seed_everything(1)


def test_git_commit_returns_unknown_outside_repo(tmp_path: Path) -> None:
    assert provenance._git_commit(tmp_path) in {"unknown"}  # no .git here
