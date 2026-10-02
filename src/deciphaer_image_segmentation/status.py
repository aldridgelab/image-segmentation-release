from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from deciphaer_image_segmentation import skiplist
from deciphaer_image_segmentation.config import PipelineConfig


@dataclass
class StageStatus:
    name: str
    state: str
    progress: str
    detail: str = ""


def squeue_states(job_ids: Iterable[str]) -> dict[str, str]:
    ids = [j for j in job_ids if j]
    if not ids:
        return {}
    if shutil.which("squeue") is None:
        return {}
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T %R"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return {}
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) >= 2:
            states[parts[0]] = parts[1] + (f" ({parts[2]})" if len(parts) == 3 else "")
    return states


def _count_tifs(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for entry in os.scandir(directory) if entry.is_file() and entry.name.lower().endswith((".tif", ".tiff")))


def _batch_state(
    config: PipelineConfig,
    batch_name: str,
    slurm_state: str | None,
    job_id: str | None,
    log_dir: Path | None,
) -> StageStatus:
    expected = _count_tifs(config.batch_dir / batch_name)
    masks_dir = config.momia_batches_dir / batch_name / "masks"
    done_count = _count_tifs(masks_dir)
    summary = config.momia_batches_dir / batch_name / "run_summary.json"
    detail = ""
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            failed = int(data.get("failed", 0))
            if failed == 0:
                state = "done"
            else:
                state = f"done ({failed} failed)"
                errors = _summarise_failed_results(data.get("results", []))
                if errors:
                    detail = errors
        except (json.JSONDecodeError, OSError):
            state = "done?"
    elif slurm_state:
        state = slurm_state.lower()
    elif done_count > 0:
        # No summary but masks exist → run died mid-batch. Try to read the slurm
        # log for the actual reason (OOM, traceback, etc.) so the user sees it.
        reason = diagnose_slurm_log(log_dir, job_id) if job_id else ""
        missing = _missing_image_ids(config.batch_dir / batch_name, masks_dir)
        if missing:
            display = ", ".join(missing[:3]) + (f" (+{len(missing) - 3} more)" if len(missing) > 3 else "")
            detail = f"missing: {display}"
            if reason:
                detail += f"  ({reason})"
        elif reason:
            detail = reason
        state = "interrupted"
    else:
        state = "pending"
    progress = f"{done_count}/{expected}" if expected else f"{done_count}"
    return StageStatus(name=batch_name, state=state, progress=progress, detail=detail)


def _missing_image_ids(batch_input: Path, masks_dir: Path) -> list[str]:
    if not batch_input.is_dir():
        return []
    existing = {p.name.removesuffix("_mask.tif") for p in masks_dir.glob("*_mask.tif")} if masks_dir.exists() else set()
    missing: list[str] = []
    for symlink in sorted(batch_input.glob("*")):
        if not symlink.is_symlink():
            continue
        if symlink.stem not in existing:
            missing.append(symlink.stem)
    return missing


def _summarise_failed_results(results: list[dict]) -> str:
    failures = [r for r in results if isinstance(r, dict) and r.get("status") == "failed"]
    if not failures:
        return ""
    sample = failures[0]
    image = sample.get("image_id", "?")
    err = (sample.get("error") or "").splitlines()[0][:120]
    extra = f" (+{len(failures) - 1} more)" if len(failures) > 1 else ""
    return f"{image}: {err}{extra}"


def diagnose_slurm_log(log_dir: Path | None, job_id: str) -> str:
    if not log_dir or not log_dir.is_dir():
        return ""
    # New layout writes <name>-<jobid>.log and <name>-<jobid>.err; older runs
    # used a single combined <name>-<jobid>.out. Prefer .err for failure
    # diagnosis, fall back to .log / .out.
    candidates = (
        list(log_dir.glob(f"*-{job_id}.err"))
        + list(log_dir.glob(f"*-{job_id}.log"))
        + list(log_dir.glob(f"*-{job_id}.out"))
        + list(log_dir.glob(f"slurm-{job_id}.out"))
    )
    if not candidates:
        return ""
    log = candidates[0]
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if "oom_kill" in text.lower() or "out of memory" in text.lower() or "OOM" in text:
        return f"OOM-killed (see {log.name})"
    for marker in ("Traceback (most recent call last)", "Error:", "ERROR"):
        idx = text.rfind(marker)
        if idx >= 0:
            tail = text[idx:].splitlines()
            for line in reversed(tail):
                line = line.strip()
                if line and not line.startswith("File "):
                    return f"{line[:140]} (see {log.name})"
    return ""


def _stage_done(config: PipelineConfig, stage: str) -> bool:
    if stage == "compile":
        return (config.momia_compiled_dir / "cell_measurements.csv").exists()
    if stage == "filter":
        return config.filter.enabled and (config.filter_dir / "cell_measurements.csv").exists()
    if stage == "crop":
        if (config.crops_dir / "crops_manifest.csv").exists():
            return True
        n = config.unet.crop_shards if config.unet.enabled else 1
        if n > 1:
            return all(
                (config.crops_dir / f"crops_manifest_shard_{k}_of_{n}.csv").exists()
                for k in range(n)
            )
        return False
    if stage == "unet":
        return (config.unet_dir / "unet_classifications.csv").exists()
    if stage == "final":
        return (config.final_dir / "extract_summary.json").exists()
    return False


def collect_status(config: PipelineConfig, *, log_dir: Path | None = None) -> dict[str, list[StageStatus]]:
    manifest_path = config.output_dir / "slurm_submission_manifest.json"
    submitted: dict[str, str] = {}
    if manifest_path.exists():
        try:
            submitted = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            submitted = {}
    slurm_states = squeue_states(submitted.values())

    prefix = config.slurm.job_prefix
    batches: list[StageStatus] = []
    for batch_name in config.batch_names:
        job_id = submitted.get(f"{prefix}_{batch_name}_momia")
        slurm_state = slurm_states.get(job_id) if job_id else None
        batches.append(_batch_state(config, batch_name, slurm_state, job_id, log_dir))

    unet_shards: list[StageStatus] = _unet_shard_statuses(config, submitted, slurm_states)

    downstream: list[StageStatus] = []
    for stage in ("compile", "filter", "crop", "unet", "final"):
        if stage == "filter" and not config.filter.enabled:
            continue
        if stage in ("crop", "unet") and not config.unet.enabled:
            continue
        if stage == "unet" and unet_shards:
            done_n = sum(1 for s in unet_shards if s.state.startswith("done"))
            total_n = len(unet_shards)
            if done_n == total_n:
                state = "done"
            elif any("running" in s.state for s in unet_shards):
                state = f"running ({done_n}/{total_n} shards done)"
            elif any(s.state == "interrupted" for s in unet_shards):
                state = f"interrupted ({done_n}/{total_n} shards done)"
            else:
                state = f"pending ({done_n}/{total_n} shards done)"
            downstream.append(StageStatus(name=stage, state=state, progress="", detail=""))
            continue
        if stage == "crop" and config.unet.crop_shards > 1:
            n = config.unet.crop_shards
            shard_csvs = [config.crops_dir / f"crops_manifest_shard_{k}_of_{n}.csv" for k in range(n)]
            shard_jobs = [submitted.get(f"{prefix}_crop_{k}_of_{n}") for k in range(n)]
            shard_states = [slurm_states.get(j) if j else None for j in shard_jobs]
            done_n = sum(1 for csv in shard_csvs if csv.exists())
            if _stage_done(config, "crop"):
                state = "done"
            elif any(s and "running" in s.lower() for s in shard_states):
                state = f"running ({done_n}/{n} shards done)"
            else:
                state = f"pending ({done_n}/{n} shards done)"
            downstream.append(StageStatus(name=stage, state=state, progress="", detail=""))
            continue
        job_id = submitted.get(f"{prefix}_{stage}") or submitted.get(f"{prefix}_unet_0_of_{config.unet.shards}")
        slurm_state = slurm_states.get(job_id) if job_id else None
        if _stage_done(config, stage):
            state = "done"
        elif slurm_state:
            state = slurm_state.lower()
        else:
            state = "pending"
        downstream.append(StageStatus(name=stage, state=state, progress="", detail=job_id or ""))

    return {"batches": batches, "downstream": downstream, "unet_shards": unet_shards}


def _unet_shard_statuses(
    config: PipelineConfig,
    submitted: dict[str, str],
    slurm_states: dict[str, str],
) -> list[StageStatus]:
    if not config.unet.enabled or config.unet.shards <= 1:
        return []
    prefix = config.slurm.job_prefix
    n = config.unet.shards
    total = _count_tifs(config.crops_dir)
    statuses: list[StageStatus] = []
    for k in range(n):
        expected = (total + (n - 1 - k)) // n if total else 0
        final = config.unet_dir / f"unet_classifications_shard_{k}_of_{n}.csv"
        partial = final.parent / (final.name + ".partial")
        job_id = submitted.get(f"{prefix}_unet_{k}_of_{n}")
        slurm_state = slurm_states.get(job_id) if job_id else None

        if final.exists():
            state = "done"
            done = expected
        elif partial.exists():
            done = max(0, _count_lines(partial) - 1)
            if slurm_state:
                state = slurm_state.lower()
            else:
                state = "interrupted"
        else:
            done = 0
            state = slurm_state.lower() if slurm_state else "pending"

        progress = f"{done}/{expected}" if expected else f"{done}"
        statuses.append(StageStatus(name=f"unet {k}/{n}", state=state, progress=progress, detail=job_id or ""))
    return statuses


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def render_status(config: PipelineConfig, *, log_dir: Path | None = None, skip_path: Path | None = None) -> str:
    payload = collect_status(config, log_dir=log_dir)
    batches: list[StageStatus] = payload["batches"]
    downstream: list[StageStatus] = payload["downstream"]
    unet_shards: list[StageStatus] = payload.get("unet_shards", [])
    total = len(batches)
    done = sum(1 for b in batches if b.state.startswith("done"))
    running = sum(1 for b in batches if "running" in b.state)
    pending = sum(1 for b in batches if b.state == "pending")
    interrupted = sum(1 for b in batches if b.state == "interrupted")

    lines: list[str] = []
    lines.append(f"Pipeline: {config.run.name}")
    lines.append(f"Output:   {config.output_dir}")
    lines.append("")
    lines.append(
        f"MOMIA batches: {done}/{total} done"
        + (f", {running} running" if running else "")
        + (f", {pending} pending" if pending else "")
        + (f", {interrupted} interrupted" if interrupted else "")
    )
    width = max((len(b.name) for b in batches), default=0)
    pwidth = max((len(b.progress) for b in batches), default=0)
    for batch in batches:
        marker = {
            "done": "✓",
            "pending": " ",
            "interrupted": "!",
        }.get(batch.state.split()[0], "·")
        suffix = f"   {batch.detail}" if batch.detail else ""
        lines.append(f"  [{marker}] {batch.name.ljust(width)}  {batch.progress.rjust(pwidth)}  {batch.state}{suffix}")
    lines.append("")
    lines.append("Downstream stages:")
    swidth = max((len(s.name) for s in downstream), default=0)
    for stage in downstream:
        marker = "✓" if stage.state == "done" else (" " if stage.state.startswith("pending") else "·")
        suffix = f"  [{stage.detail}]" if stage.detail else ""
        lines.append(f"  [{marker}] {stage.name.ljust(swidth)}  {stage.state}{suffix}")
    if unet_shards:
        n = len(unet_shards)
        sdone = sum(1 for s in unet_shards if s.state.startswith("done"))
        srunning = sum(1 for s in unet_shards if "running" in s.state)
        spending = sum(1 for s in unet_shards if s.state == "pending")
        sinterrupted = sum(1 for s in unet_shards if s.state == "interrupted")
        lines.append("")
        lines.append(
            f"U-Net shards: {sdone}/{n} done"
            + (f", {srunning} running" if srunning else "")
            + (f", {spending} pending" if spending else "")
            + (f", {sinterrupted} interrupted" if sinterrupted else "")
        )
        nwidth = max((len(s.name) for s in unet_shards), default=0)
        pwidth = max((len(s.progress) for s in unet_shards), default=0)
        for shard in unet_shards:
            head = shard.state.split()[0] if shard.state else ""
            marker = {"done": "✓", "pending": " ", "interrupted": "!"}.get(head, "·")
            suffix = f"   {shard.detail}" if shard.detail else ""
            lines.append(f"  [{marker}] {shard.name.ljust(nwidth)}  {shard.progress.rjust(pwidth)}  {shard.state}{suffix}")
    blocked = [b for b in batches if b.state == "interrupted"] + [
        s for s in downstream if "dependencyneversatisfied" in s.state.lower()
    ]
    if blocked:
        lines.append("")
        lines.append("Action: re-run `pipeline.py --config <yaml>` to resubmit interrupted/blocked stages.")
        lines.append("        (failed images are auto-flagged into skip.txt; edit it before re-running to retry any.)")
    if skip_path is not None:
        skip_tokens = sorted(skiplist.load(skip_path))
        if skip_tokens:
            lines.append("")
            lines.append(f"Skip list ({len(skip_tokens)} entries — {skip_path}):")
            for token in skip_tokens[:20]:
                lines.append(f"  - {token}")
            if len(skip_tokens) > 20:
                lines.append(f"  ... +{len(skip_tokens) - 20} more")
    return "\n".join(lines)
