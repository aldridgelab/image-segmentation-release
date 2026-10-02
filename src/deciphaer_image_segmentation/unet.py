from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import tifffile

from tb_unet.inference import TBUNetPredictor

_log = logging.getLogger(__name__)


def classify_crops(
    *,
    crops_dir: Path,
    output_csv: Path,
    model_path: Path,
    device: str | None,
    probability_threshold: float,
    use_mask_channel: bool,
    shard: tuple[int, int] | None = None,
) -> Path:
    if not model_path.is_file():
        raise FileNotFoundError(f"U-Net model checkpoint not found: {model_path}")

    paths = sorted(crops_dir.glob("*.tif"))
    total_crops = len(paths)
    if shard is not None:
        k, n = shard
        paths = paths[k::n]
        _log.info("unet: shard %d/%d takes %d of %d crops", k, n, len(paths), total_crops)
    _log.info(
        "unet: loading model %s on device=%s (use_mask_channel=%s, threshold=%.2f)",
        model_path,
        device or "auto",
        use_mask_channel,
        probability_threshold,
    )
    predictor = TBUNetPredictor(checkpoint_path=model_path, device=device)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling .partial file and rename on success so a preempted/killed
    # shard never leaves the canonical filename in place — _stage_done would otherwise
    # treat the truncated CSV as complete and merge_classification_shards would silently
    # produce a short final dataset.
    partial_csv = output_csv.parent / (output_csv.name + ".partial")
    fieldnames = [
        "Cell_ID",
        "crop_path",
        "predicted_class",
        "confidence",
        "p_background",
        "p_cell",
        "p_noncell",
        "predicted_cell",
    ]
    counts = {"background": 0, "cell": 0, "non_cell": 0}
    log_step = max(1, len(paths) // 10)
    # Line-buffered so each writerow flushes to the OS — otherwise the 8 KB
    # default text buffer only hits disk every ~50 rows, and the --status
    # dashboard (which counts lines in this .partial) gets stuck at 0 for
    # minutes at a time on slow CPU/NFS runs.
    with partial_csv.open("w", newline="", encoding="utf-8", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, crop_path in enumerate(paths, start=1):
            stack, mask = _load_crop(crop_path)
            roi_mask = mask if use_mask_channel else None
            _, probs = predictor.predict(stack, return_probs=True)
            p_bg, p_cell, p_nc = _class_probs(probs=probs, roi_mask=roi_mask)
            class_idx = int(np.argmax([p_bg, p_cell, p_nc]))
            class_name = ["background", "cell", "non_cell"][class_idx]
            confidence = [p_bg, p_cell, p_nc][class_idx]
            counts[class_name] += 1
            writer.writerow(
                {
                    "Cell_ID": crop_path.stem,
                    "crop_path": str(crop_path),
                    "predicted_class": class_name,
                    "confidence": f"{confidence:.6f}",
                    "p_background": f"{p_bg:.6f}",
                    "p_cell": f"{p_cell:.6f}",
                    "p_noncell": f"{p_nc:.6f}",
                    "predicted_cell": bool(class_name == "cell" and p_cell >= probability_threshold),
                }
            )
            if idx % log_step == 0:
                _log.info("unet: classified %d/%d crops", idx, len(paths))
    partial_csv.replace(output_csv)
    _log.info(
        "unet: done — %d cell, %d non_cell, %d background; wrote %s",
        counts["cell"],
        counts["non_cell"],
        counts["background"],
        output_csv,
    )
    return output_csv


def merge_classification_shards(unet_dir: Path, *, n_shards: int) -> Path:
    frames = []
    import pandas as pd

    for k in range(n_shards):
        path = unet_dir / f"unet_classifications_shard_{k}_of_{n_shards}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No U-Net classification shards found in {unet_dir}")
    merged = pd.concat(frames, ignore_index=True)
    out = unet_dir / "unet_classifications.csv"
    merged.to_csv(out, index=False)
    return out


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    left, right = spec.split("/", 1)
    k, n = int(left), int(right)
    if n < 1 or k < 0 or k >= n:
        raise ValueError(f"Invalid shard spec {spec!r}; expected 0 <= K < N")
    return k, n


def _load_crop(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        return arr[None], None
    if arr.ndim != 3:
        raise ValueError(f"Unexpected crop shape {arr.shape} in {path}")
    mask = arr[-1] if arr.shape[0] >= 2 else None
    return arr, mask


def _class_probs(*, probs: np.ndarray, roi_mask: np.ndarray | None) -> tuple[float, float, float]:
    if roi_mask is not None and roi_mask.any():
        selected = roi_mask > 0
        return float(probs[0][selected].mean()), float(probs[1][selected].mean()), float(probs[2][selected].mean())
    _, height, width = probs.shape
    cy, cx = height // 2, width // 2
    margin = max(1, min(height, width) // 4)
    window = np.s_[cy - margin : cy + margin, cx - margin : cx + margin]
    return float(probs[0][window].mean()), float(probs[1][window].mean()), float(probs[2][window].mean())
