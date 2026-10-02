from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        from loguru import logger
        logger.debug("torch not installed; skipping torch seeding for seed={}", seed)
        return
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        from loguru import logger
        logger.exception("torch seeding failed for seed={}; continuing without torch determinism", seed)


def write_run_metadata(
    *,
    output_dir: Path,
    config_snapshot: dict[str, Any],
    input_paths: list[str],
    seed: int,
    artifacts: dict[str, str],
    run_hash: str | None = None,
) -> Path:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "segmentation_run_hash": run_hash or "",
        "input_paths": input_paths,
        "config": _redact(config_snapshot),
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(["numpy", "pandas", "PyYAML", "tifffile"]),
        "seed": seed,
        "artifacts": artifacts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_commit() -> str:
    checkout_root = Path(__file__).resolve().parents[2]
    try:
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=checkout_root,
            check=True,
            capture_output=True,
            text=True,
        )
        if Path(repo_root.stdout.strip()).resolve() != checkout_root:
            return "unknown"
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("secret", "password", "token", "api_key")):
                out[key] = "<redacted>"
            else:
                out[key] = _redact(item)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
