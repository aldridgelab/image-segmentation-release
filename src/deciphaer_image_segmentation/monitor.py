"""Sidecar monitor: warn when a MOMIA batch stalls.

Runs in parallel with the rest of the slurm DAG (no dependencies, tiny
resources). On a fixed poll interval it inspects each MOMIA batch's
mask count vs. its slurm state. A batch is considered "stalled" when:

* slurm says it's still RUNNING, AND
* the count of ``*_mask.tif`` files in the batch's output dir hasn't
  increased in ``stall_minutes`` of wall-clock time.

For each stall it posts a single webhook (deduped via a state file in
the run's output dir). The pipeline keeps running — the watchdog at
compile time still does the actual flag-and-continue. This monitor is
just for *proactive* notification: "heads up, batch14 looks stuck",
so the user can investigate or scancel before the slurm wall hits.

Exits cleanly when no MOMIA batch is still RUNNING/PENDING in the
manifest.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from deciphaer_image_segmentation import notify
from deciphaer_image_segmentation.config import PipelineConfig
from deciphaer_image_segmentation.status import squeue_states

_log = logging.getLogger(__name__)


def run_monitor(
    config: PipelineConfig,
    *,
    stall_minutes: float = 5.0,
    poll_seconds: float = 60.0,
    max_runtime_seconds: float = 8 * 3600,
) -> Path:
    """Loop until all MOMIA batches have terminated; warn on stalls.

    Returns the path to the persisted state file.
    """
    output_dir = config.output_dir
    state_path = output_dir / ".monitor_state.json"
    manifest_path = output_dir / "slurm_submission_manifest.json"

    if not manifest_path.exists():
        _log.warning("monitor: no slurm manifest at %s — exiting", manifest_path)
        return state_path

    state = _load_state(state_path)
    started = time.monotonic()
    stall_seconds = float(stall_minutes) * 60.0
    poll_seconds = max(5.0, float(poll_seconds))
    webhook_url = config.notifications.webhook_url
    prefix = config.slurm.job_prefix

    _log.info(
        "monitor: poll every %.0fs, stall threshold %.1f min, output_dir=%s",
        poll_seconds, stall_minutes, output_dir,
    )

    while time.monotonic() - started < max_runtime_seconds:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
        except (OSError, json.JSONDecodeError):
            time.sleep(poll_seconds)
            continue

        # Map of <name> -> <slurm job id> for MOMIA batches only.
        batch_jobs: dict[str, str] = {}
        for name, job_id in manifest.items():
            if name.endswith("_momia") and job_id:
                batch_jobs[name] = str(job_id)

        if not batch_jobs:
            _log.info("monitor: no MOMIA batches in manifest — exiting")
            _save_state(state_path, state)
            return state_path

        states = squeue_states(list(batch_jobs.values()))
        active = 0  # batches still PENDING or RUNNING
        now = time.time()

        for name, job_id in batch_jobs.items():
            slurm_state = states.get(job_id, "")
            head = slurm_state.split()[0] if slurm_state else ""
            if head in ("PENDING", "RUNNING"):
                active += 1

            if not head.startswith("RUNNING"):
                # Only check progress for batches that should be making any.
                continue

            batch_name = name.removeprefix(f"{prefix}_").removesuffix("_momia")
            masks_dir = config.momia_batches_dir / batch_name / "masks"
            current = (
                sum(1 for _ in masks_dir.glob("*_mask.tif"))
                if masks_dir.is_dir() else 0
            )

            entry = state.get(name, {})
            last_count = entry.get("last_count")
            last_change_at = entry.get("last_change_at")
            warned = entry.get("warned", False)

            if last_count is None or current > int(last_count):
                # First sighting or progress made: reset the clock.
                state[name] = {
                    "last_count": int(current),
                    "last_change_at": float(now),
                    "warned": False,
                    "job_id": job_id,
                }
                continue

            stalled_for = now - float(last_change_at)
            if stalled_for >= stall_seconds and not warned:
                _emit_stall_warning(
                    webhook_url=webhook_url,
                    run_name=config.run.name,
                    batch_name=batch_name,
                    job_id=job_id,
                    mask_count=int(current),
                    stalled_for_minutes=stalled_for / 60.0,
                )
                entry["warned"] = True
                state[name] = entry

        _save_state(state_path, state)

        if active == 0:
            _log.info("monitor: all MOMIA batches terminated — exiting")
            return state_path

        time.sleep(poll_seconds)

    _log.warning("monitor: max runtime (%ss) reached — exiting", max_runtime_seconds)
    _save_state(state_path, state)
    return state_path


def _emit_stall_warning(
    *,
    webhook_url: str | None,
    run_name: str,
    batch_name: str,
    job_id: str,
    mask_count: int,
    stalled_for_minutes: float,
) -> None:
    msg = (
        f":hourglass_flowing_sand: *{run_name}*: `{batch_name}` looks stalled — "
        f"{mask_count} mask(s) produced; no new output in "
        f"{stalled_for_minutes:.1f} min (slurm job `{job_id}`). "
        "Pipeline still running; if it's hung, `scancel` it and the watchdog "
        "will skip the missing images and proceed."
    )
    _log.warning(
        "monitor: stall detected — %s (job %s) at %d masks, %.1f min idle",
        batch_name, job_id, mask_count, stalled_for_minutes,
    )
    if webhook_url:
        notify.post(webhook_url, msg)


def _load_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass
