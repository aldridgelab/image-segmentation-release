from __future__ import annotations

import json
import random
import subprocess
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


def test_write_run_metadata_uses_software_checkout_not_output_repo(tmp_path: Path) -> None:
    output_dir = tmp_path / "unrelated-repo"
    output_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=output_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=output_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=output_dir, check=True)
    (output_dir / "unrelated.txt").write_text("unrelated", encoding="utf-8")
    subprocess.run(["git", "add", "unrelated.txt"], cwd=output_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=output_dir, check=True)

    checkout_root = Path(provenance.__file__).resolve().parents[2]
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    path = write_run_metadata(
        output_dir=output_dir,
        config_snapshot={},
        input_paths=[],
        seed=1,
        artifacts={},
    )

    assert json.loads(path.read_text(encoding="utf-8"))["git_commit"] == expected_commit


def test_git_commit_returns_unknown_when_package_has_no_checkout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        provenance,
        "__file__",
        str(tmp_path / "site-packages" / "deciphaer_image_segmentation" / "provenance.py"),
    )

    assert provenance._git_commit() == "unknown"
