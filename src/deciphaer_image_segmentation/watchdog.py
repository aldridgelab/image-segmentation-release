"""Post-mortem failure handling for MOMIA batch jobs.

Used by:

* The compile stage as a watchdog. Compile is wired to slurm with
  ``afterany`` on every MOMIA batch, so it always runs once each batch
  has finished (success or fail). It calls :func:`flag_failed_batches`
  which detects failures, posts a webhook, and appends the affected
  images to ``skip.txt`` so a re-submit doesn't redo them.

* The runner's resume path (``_auto_flag_failed_images``). When a user
  re-runs ``--submit`` after a partial failure, the same logic catches
  any failed batches that the watchdog couldn't see (e.g. local-mode
  runs without a slurm manifest, or older runs).

Detection is intentionally cheap and read-only: it inspects
``run_summary.json``, the symlinks in each batch's input dir, and the
slurm log file (via :func:`status.diagnose_slurm_log`) if present.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from deciphaer_image_segmentation import notify, skiplist
from deciphaer_image_segmentation.config import PipelineConfig
from deciphaer_image_segmentation.status import diagnose_slurm_log, squeue_states

_log = logging.getLogger(__name__)


def flag_failed_batches(
    config: PipelineConfig,
    *,
    log_dir: Path | None,
    skip_path: Path,
    webhook_url: str | None,
    manifest_path: Path | None = None,
    context: str = "watchdog",
    skip_pending_or_running: bool = True,
) -> list[tuple[str, str]]:
    """Detect failed MOMIA batches, flag their images, and post a webhook.

    Returns a list of ``(image_id, reason)`` for everything that was
    auto-flagged; empty list if nothing failed.

    Parameters
    ----------
    config:
        Pipeline config (provides batch names, output paths, run name).
    log_dir:
        Slurm log subdir for this submission. Used to read ``.err`` /
        ``.log`` files for OOM/traceback diagnosis. ``None`` skips log
        diagnosis.
    skip_path:
        Path to ``skip.txt`` — failed images get appended here.
    webhook_url:
        Slack-compatible webhook. Skipped if ``None``.
    manifest_path:
        Path to ``slurm_submission_manifest.json``. If absent, we still
        scan run_summary.json + masks dir (e.g. local runs).
    context:
        Free-form label included in webhook + log messages so the user
        knows whether the watchdog or resume path flagged them.
    skip_pending_or_running:
        When called from the resume path we want to leave still-running
        jobs alone; when called from compile (which runs after batches
        terminate via ``afterany``) we don't have to ignore them, but
        the squeue check is cheap and we keep the same conservative
        behavior.
    """
    submitted: dict[str, str] = {}
    if manifest_path is None:
        manifest_path = config.output_dir / "slurm_submission_manifest.json"
    if manifest_path.exists():
        try:
            submitted = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            submitted = {}

    states: dict[str, str] = {}
    if submitted and skip_pending_or_running:
        states = squeue_states(list(submitted.values()))

    prefix = config.slurm.job_prefix
    flagged: list[tuple[str, str]] = []

    for batch_name in config.batch_names:
        job_name = f"{prefix}_{batch_name}_momia"
        job_id = submitted.get(job_name) if submitted else None
        if (
            job_id
            and skip_pending_or_running
            and states.get(job_id, "").startswith(("PENDING", "RUNNING"))
        ):
            continue

        batch_input = config.batch_dir / batch_name
        masks_dir = config.momia_batches_dir / batch_name / "masks"
        if not batch_input.is_dir():
            continue

        # Per-image failures recorded by MOMIA (typed errors, not crashes).
        per_image_failures: dict[str, str] = {}
        summary_path = config.momia_batches_dir / batch_name / "run_summary.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = {}
            for result in payload.get("results", []) or []:
                if isinstance(result, dict) and result.get("status") == "failed":
                    image_id = str(result.get("image_id", ""))
                    if image_id:
                        per_image_failures[image_id] = (
                            (result.get("error") or "MOMIA error").splitlines()[0][:160]
                        )

        # Anything in the symlink set without a corresponding mask is missing.
        existing_masks = (
            {p.name.removesuffix("_mask.tif") for p in masks_dir.glob("*_mask.tif")}
            if masks_dir.exists() else set()
        )
        for symlink in batch_input.glob("*"):
            if not symlink.is_symlink():
                continue
            image_id = symlink.stem
            if image_id in existing_masks:
                continue
            reason = (
                per_image_failures.get(image_id)
                or (diagnose_slurm_log(log_dir, job_id) if (log_dir and job_id) else "")
                or "missing after batch finished"
            )
            flagged.append((image_id, f"{batch_name} (job {job_id or '?'}): {reason}"))

    if not flagged:
        return []

    # Append-only: skiplist.append already de-dupes against existing entries.
    new_entries = [image_id for image_id, _ in flagged]
    skiplist.append(
        skip_path,
        new_entries,
        comment=f"auto-flagged by {context} ({len(new_entries)} image(s))",
    )
    for image_id, why in flagged:
        _log.warning("Auto-flagged %s — %s", image_id, why)
    _log.warning(
        "Added %d image(s) to %s. They will be skipped on the next run; edit and re-run to retry any.",
        len(new_entries),
        skip_path,
    )

    if webhook_url:
        lines = [
            f":warning: *{config.run.name}*: {context} flagged {len(flagged)} image(s) "
            "(added to `skip.txt`):"
        ]
        for image_id, why in flagged[:8]:
            lines.append(f"• `{image_id}` — {why}")
        if len(flagged) > 8:
            lines.append(f"• …+{len(flagged) - 8} more")
        notify.post(webhook_url, "\n".join(lines))

    return flagged
