"""Sample N random files from a directory into a new directory.

Usage:
    uv run python scripts/sample_crops.py <src_dir> <dst_dir> [-n 200] [--seed 0] [--ext tif]

Example:
    uv run python scripts/sample_crops.py \
        /path/to/results/final/passed_crops \
        /path/to/sample_crops -n 200
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="Source directory.")
    parser.add_argument("dst", type=Path, help="Destination directory (created if missing).")
    parser.add_argument("-n", "--num", type=int, default=200, help="Number of files to sample (default: 200).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility (default: 0).")
    parser.add_argument("--ext", default="tif", help="File extension to filter on, no dot (default: tif).")
    parser.add_argument("--move", action="store_true", help="Move instead of copy.")
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"error: source directory not found: {args.src}", file=sys.stderr)
        return 1

    pattern = f"*.{args.ext.lstrip('.')}"
    candidates = [p for p in args.src.iterdir() if p.is_file() and p.match(pattern)]
    if not candidates:
        print(f"error: no '{pattern}' files in {args.src}", file=sys.stderr)
        return 1

    n = min(args.num, len(candidates))
    random.seed(args.seed)
    sample = random.sample(candidates, n)

    args.dst.mkdir(parents=True, exist_ok=True)
    op = shutil.move if args.move else shutil.copy2
    for src_path in sample:
        op(str(src_path), str(args.dst / src_path.name))

    verb = "moved" if args.move else "copied"
    print(f"{verb} {n} of {len(candidates)} '{pattern}' files from {args.src} -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
