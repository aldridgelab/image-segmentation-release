#!/usr/bin/env python3
"""
Submit TB U-Net experiment batches to SLURM via sbatch.

Splits a multi-experiment YAML config into N batches, generates per-batch
YAML files and SBATCH scripts, submits them, and records a session manifest.

Example usage:
    # Generate sweep configs first
    python tb_unet/scripts/generate_sweep_config.py --template tb_unet/configs/sweeps/sweep_params.yaml --method grid

    # Submit all 216 runs across 20 batches (GPU)
    python tb_unet/scripts/submit_sbatch.py \
        --config tb_unet/configs_generated/unet_sweep_v1_bundle.yaml \
        --num-batches 20 \
        --use-gpu

    # Dry run (preview without submitting)
    python tb_unet/scripts/submit_sbatch.py \
        --config tb_unet/configs_generated/unet_sweep_v1_bundle.yaml \
        --num-batches 20 \
        --use-gpu \
        --dry-run

Each job runs:
    python -m tb_unet.train --config <batch_yaml>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def load_yaml(path: Path) -> Any:
    """Load YAML file."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: Any, path: Path) -> None:
    """Save YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def split_items(items: list[Any], num_batches: int) -> list[list[Any]]:
    """Split items into N batches as evenly as possible."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    n = len(items)
    if n == 0:
        return [[] for _ in range(num_batches)]

    # Distribute items evenly
    base = n // num_batches
    remainder = n % num_batches
    batches: list[list[Any]] = []
    start = 0

    for i in range(num_batches):
        # Earlier batches get one extra item if there's a remainder
        size = base + (1 if i < remainder else 0)
        end = start + size
        if size > 0:
            batches.append(items[start:end])
        start = end

    return batches


def determine_structure(data: Any) -> tuple[str, list[Any]]:
    """
    Determine how experiments are stored in YAML.

    Returns (kind, items) where kind is:
    - 'experiments': top-level dict with key 'experiments': list
    - 'list': top-level list
    """
    if isinstance(data, dict) and "experiments" in data and isinstance(data["experiments"], list):
        return ("experiments", data["experiments"])
    if isinstance(data, list):
        return ("list", data)
    raise ValueError("Unsupported YAML format. Expected 'experiments' key or top-level list.")


def rebuild_yaml_subset(kind: str, subset: list[Any]) -> Any:
    """Rebuild YAML structure from subset of items."""
    if kind == "experiments":
        return {"experiments": subset}
    elif kind == "list":
        return subset
    else:
        raise ValueError(f"Unknown kind: {kind}")


def make_sbatch_script(
    out_path: Path,
    job_name: str,
    partition: str,
    time_limit: str,
    cpus_per_task: int,
    mem: str,
    gpus: int,
    gpu_flag_style: str,
    account: str | None,
    log_dir: Path,
    setup_cmd: str | None,
    python_cmd: str,
    main_args: list[str],
) -> None:
    """Generate SBATCH script file."""
    lines: list[str] = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH -o {log_dir}/%x_%j.out",
        f"#SBATCH -e {log_dir}/%x_%j.err",
        f"#SBATCH -D {os.getcwd()}",
    ]

    if account:
        lines.append(f"#SBATCH -A {account}")

    if gpus > 0:
        if gpu_flag_style == "gres":
            lines.append(f"#SBATCH --gres=gpu:{gpus}")
        else:
            lines.append(f"#SBATCH --gpus={gpus}")

    lines.extend([
        "",
        "set -euo pipefail",
        'echo "Running on $(hostname)"',
        'echo "Start: $(date -Is)"',
        "",
    ])

    if setup_cmd:
        lines.extend([setup_cmd, ""])

    cmd = " ".join([python_cmd] + main_args)
    lines.append(cmd)
    lines.extend(["", 'echo "End: $(date -Is)"'])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def submit_sbatch(script_path: Path) -> tuple[int, str]:
    """Submit sbatch script and return (job_id, raw_output)."""
    proc = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
    stdout = proc.stdout.strip()
    # Typical: "Submitted batch job 123456"
    m = re.search(r"(\d+)$", stdout)
    job_id = int(m.group(1)) if m else -1
    return job_id, stdout


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Submit TB U-Net batches via sbatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Submit with GPU (default profile: A100, 32G, 24h)
    python tb_unet/scripts/submit_sbatch.py \\
        --config tb_unet/configs_generated/unet_sweep_v1_bundle.yaml \\
        --num-batches 20 \\
        --use-gpu

    # Dry run (preview without submitting)
    python tb_unet/scripts/submit_sbatch.py \\
        --config tb_unet/configs_generated/unet_sweep_v1_bundle.yaml \\
        --num-batches 20 \\
        --dry-run

    # Custom resources
    python tb_unet/scripts/submit_sbatch.py \\
        --config tb_unet/configs_generated/unet_sweep_v1_bundle.yaml \\
        --num-batches 20 \\
        --use-gpu \\
        --partition gpu-large \\
        --mem 64G \\
        --time 48:00:00
        """,
    )

    p.add_argument("--config", required=True, help="Path to multi-experiment YAML config")
    p.add_argument("--num-batches", type=int, default=10, help="Number of batches to split into")
    p.add_argument("--use-gpu", action="store_true", help="Use GPU defaults (SLURM resources)")

    # SLURM options (optional overrides)
    p.add_argument("--partition", default=None, help="SLURM partition (defaults: gpu/batch)")
    p.add_argument("--account", default=None, help="SLURM account (optional)")
    p.add_argument("--time", dest="time_limit", default=None, help="Time limit HH:MM:SS")
    p.add_argument("--cpus-per-task", type=int, default=None, help="CPUs per task")
    p.add_argument("--mem", default=None, help="Memory per job, e.g., 32G")
    p.add_argument("--gpus", type=int, default=None, help="GPUs per job")
    p.add_argument(
        "--gpu-flag-style",
        choices=["gres", "gpus"],
        default=None,
        help="Use --gres=gpu:X or --gpus=X (default: gres)",
    )

    # Runtime setup
    p.add_argument(
        "--python-cmd",
        default="python -m tb_unet.train",
        help="Python command to run training",
    )
    p.add_argument(
        "--setup-cmd",
        default=None,
        help="Optional shell command(s) to setup env (e.g., 'source ~/.bashrc && conda activate myenv')",
    )

    # Output/session
    p.add_argument("--job-name-prefix", default="tb_unet", help="Job name prefix")
    p.add_argument("--session-name", default=None, help="Optional session name")
    p.add_argument("--work-dir", default="tb_unet/hpc_submissions", help="Base directory for batch files")
    p.add_argument("--dry-run", action="store_true", help="Write files but do not call sbatch")

    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    """Main entry point."""
    args = parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 2

    data = load_yaml(config_path)
    kind, items = determine_structure(data)

    logger.info(f"Loaded {len(items)} experiments from {config_path}")

    # Default resource profiles
    profile_gpu = dict(
        partition="gpu",
        time_limit="24:00:00",
        cpus_per_task=4,
        mem="32G",
        gpus=1,
        gpu_flag_style="gres",
    )
    profile_cpu = dict(
        partition="batch",
        time_limit="24:00:00",
        cpus_per_task=4,
        mem="16G",
        gpus=0,
        gpu_flag_style="gres",
    )
    prof = profile_gpu if args.use_gpu else profile_cpu

    # Apply overrides or defaults
    partition = args.partition or prof["partition"]
    time_limit = args.time_limit or prof["time_limit"]
    cpus_per_task = args.cpus_per_task or prof["cpus_per_task"]
    mem = args.mem or prof["mem"]
    gpus = (args.gpus if args.gpus is not None else prof["gpus"]) if args.use_gpu else 0
    gpu_flag_style = args.gpu_flag_style or prof["gpu_flag_style"]

    # Session directory
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = config_path.stem
    session_name = args.session_name or f"{base_name}_{ts}"
    work_dir = Path(args.work_dir) / session_name
    cfg_dir = work_dir / "configs"
    sh_dir = work_dir / "sbatch"
    log_dir = work_dir / "logs"

    for d in (cfg_dir, sh_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Split into batches
    batches = split_items(items, args.num_batches)
    batches = [b for b in batches if b]  # Remove empty batches

    logger.info(f"Split into {len(batches)} batches")

    job_records: list[dict[str, Any]] = []

    for i, subset in enumerate(batches, start=1):
        # Rebuild YAML subset
        subset_yaml = rebuild_yaml_subset(kind, subset)

        batch_id = f"{i:03d}"
        batch_cfg = cfg_dir / f"batch_{batch_id}.yaml"
        dump_yaml(subset_yaml, batch_cfg)

        job_name = f"{args.job_name_prefix}_{base_name}_{batch_id}"

        # Build training command args
        main_args = ["--config", str(batch_cfg)]

        # Create sbatch script
        sbatch_path = sh_dir / f"{job_name}.sbatch"
        make_sbatch_script(
            out_path=sbatch_path,
            job_name=job_name,
            partition=partition,
            time_limit=time_limit,
            cpus_per_task=cpus_per_task,
            mem=mem,
            gpus=gpus,
            gpu_flag_style=gpu_flag_style,
            account=args.account,
            log_dir=log_dir,
            setup_cmd=args.setup_cmd,
            python_cmd=args.python_cmd,
            main_args=main_args,
        )

        submitted = {
            "batch_index": i,
            "batch_config": str(batch_cfg),
            "sbatch_script": str(sbatch_path),
            "job_name": job_name,
            "num_experiments": len(subset),
        }

        if args.dry_run:
            logger.info(f"[DRY RUN] Would submit: {sbatch_path} ({len(subset)} experiments)")
            submitted["job_id"] = None
            submitted["sbatch_stdout"] = None
        else:
            try:
                job_id, out = submit_sbatch(sbatch_path)
                logger.info(f"Submitted {job_name} -> job {job_id} ({len(subset)} experiments)")
                submitted["job_id"] = job_id
                submitted["sbatch_stdout"] = out
            except Exception as e:
                logger.error(f"Failed to submit {job_name}: {e}")
                submitted["job_id"] = None
                submitted["sbatch_stdout"] = str(e)

        job_records.append(submitted)

    # Session manifest
    manifest = {
        "session_name": session_name,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_config": str(config_path),
        "total_experiments": len(items),
        "num_batches": len(batches),
        "use_gpu": args.use_gpu,
        "partition": partition,
        "account": args.account,
        "time_limit": time_limit,
        "cpus_per_task": cpus_per_task,
        "mem": mem,
        "gpus": gpus,
        "gpu_flag_style": gpu_flag_style,
        "python_cmd": args.python_cmd,
        "setup_cmd": args.setup_cmd,
        "jobs": job_records,
    }

    manifest_path = work_dir / "session.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Wrote session manifest: {manifest_path}")

    # Summary
    logger.info("\nSubmission Summary:")
    logger.info(f"  Total experiments: {len(items)}")
    logger.info(f"  Batches: {len(batches)}")
    logger.info(f"  Partition: {partition}")
    logger.info(f"  Resources: {cpus_per_task} CPUs, {mem} RAM, {gpus} GPUs")
    logger.info(f"  Time limit: {time_limit}")
    logger.info(f"  Session dir: {work_dir}")

    if args.dry_run:
        logger.info("\n[DRY RUN] No jobs were submitted. Remove --dry-run to submit.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
