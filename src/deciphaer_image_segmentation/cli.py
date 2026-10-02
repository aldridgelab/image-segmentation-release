from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import sys
import time
from pathlib import Path


def _configure_runtime_caches() -> None:
    """Use a persistent matplotlib font cache so each Slurm job doesn't rebuild it.

    Without this MPLCONFIGDIR lands in $TMPDIR per job and matplotlib reports a few
    INFO-level "Failed to extract font" / "generated new fontManager" lines into the
    Slurm log on every cold start. Pointing it at a stable cache silences the noise
    after the first run.
    """
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache = Path.home() / ".cache" / "image-segmentation-pipeline-v2" / "matplotlib"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ["MPLCONFIGDIR"] = str(cache)


_configure_runtime_caches()


from deciphaer_image_segmentation import notify, skiplist
from deciphaer_image_segmentation.config import load_config
from deciphaer_image_segmentation.runner import PipelineRunner
from deciphaer_image_segmentation.status import render_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or submit the image segmentation pipeline v2.")
    parser.add_argument("--config", required=True, type=Path, help="Pipeline YAML config.")
    parser.add_argument("--plan", action="store_true", help="Print the Slurm DAG without submitting.")
    parser.add_argument("--submit", action="store_true", help="Set up batches and submit the Slurm DAG.")
    parser.add_argument("--status", action="store_true", help="Print MOMIA + downstream stage progress.")
    parser.add_argument("--skip", action="append", default=[], metavar="IMAGE", help="Add an image (filename, stem, or absolute path) to skip.txt and exit. Repeatable.")
    parser.add_argument("--no-resume", action="store_true", help="Submit every job even if outputs already exist.")
    parser.add_argument("--stage", choices=["setup", "momia", "compile", "filter", "crop", "unet", "final", "monitor"])
    parser.add_argument("--batch-name", help="Batch name for --stage momia, e.g. batch1.")
    parser.add_argument("--shard", help="Shard spec for --stage crop or unet, e.g. 0/16.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    logging.basicConfig(level=getattr(logging, config.run.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    # matplotlib font_manager logs font-cache rebuilds at INFO; that's noise for our use.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    runner = PipelineRunner(config)

    if args.skip:
        added = skiplist.append(runner.skip_path, args.skip, comment="manual --skip")
        if added:
            print(f"Added to {runner.skip_path}: {', '.join(added)}")
        else:
            print(f"Already present in {runner.skip_path}: {', '.join(args.skip)}")
        return 0
    if args.status:
        print(render_status(config, log_dir=runner.log_dir, skip_path=runner.skip_path))
        return 0
    if args.plan:
        _print_plan(runner)
        return 0
    if args.stage:
        log = logging.getLogger("pipeline")
        label = args.stage + (f"[{args.batch_name}]" if args.batch_name else "") + (f"[shard {args.shard}]" if args.shard else "")
        log.info("=== stage %s starting (config=%s) ===", label, args.config)
        _install_preempt_notifier(
            webhook_url=config.notifications.webhook_url,
            run_name=config.run.name,
            stage_label=label,
        )
        start = time.monotonic()
        try:
            result = runner.run_stage(args.stage, batch_name=args.batch_name, shard=args.shard)
        except Exception as exc:
            elapsed = time.monotonic() - start
            log.exception("=== stage %s FAILED after %.1fs ===", label, elapsed)
            notify.post(
                config.notifications.webhook_url,
                f":x: {config.run.name}: stage {label} failed after {elapsed:.0f}s — {type(exc).__name__}: {exc}",
            )
            raise
        elapsed = time.monotonic() - start
        log.info("=== stage %s done in %.1fs ===", label, elapsed)
        print(_jsonable(result))
        return 0
    if args.submit or config.slurm.submit or config.run.mode == "slurm":
        submitted = runner.submit(resume=not args.no_resume)
        print(json.dumps(submitted, indent=2))
        return 0
    outputs = runner.run_local()
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


def _install_preempt_notifier(*, webhook_url: str | None, run_name: str, stage_label: str) -> None:
    """Post a Slack message if SLURM signals preemption (SIGTERM) before we finish.

    SLURM sends SIGTERM at preemption time and waits GraceTime seconds before SIGKILL,
    which is enough to fire one webhook. With `--requeue` set on the job, SLURM will
    automatically resubmit the same job after preemption — atomic shard writes ensure
    the requeued attempt restarts cleanly without leaving a half-written CSV.
    """
    if not webhook_url:
        return
    if not os.environ.get("SLURM_JOB_ID"):
        return
    job_id = os.environ.get("SLURM_JOB_ID", "?")
    node = os.environ.get("SLURMD_NODENAME", "?")
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    sent = {"value": False}

    def handler(signum: int, _frame) -> None:
        if not sent["value"]:
            sent["value"] = True
            try:
                from deciphaer_image_segmentation import notify
                notify.post(
                    webhook_url,
                    f":warning: {run_name}: stage `{stage_label}` got SIGTERM (preempted/timeout) "
                    f"on `{node}` — job {job_id}, restart_count={restart}. "
                    "If --requeue is set this shard will resume automatically.",
                )
            except Exception:
                from loguru import logger
                logger.exception(
                    "preempt-notifier webhook post failed (run={}, stage={}, job={})",
                    run_name,
                    stage_label,
                    job_id,
                )
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, handler)


def _print_plan(runner: PipelineRunner) -> None:
    print(f"# config: {runner.config.config_path}")
    print(f"# output: {runner.config.output_dir}")
    for spec in runner.plan():
        deps = f" depends_on={','.join(spec.dependencies)}" if spec.dependencies else ""
        command = shlex.join([str(spec.script), *spec.args])
        print(f"{spec.name}: {command}{deps}")


def _jsonable(value) -> str:
    if isinstance(value, dict):
        return json.dumps({key: str(item) for key, item in value.items()}, indent=2)
    return str(value)
