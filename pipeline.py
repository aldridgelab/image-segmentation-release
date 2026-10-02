#!/usr/bin/env python3
"""Top-level entrypoint for the deciphaer image-segmentation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `uv run pipeline.py` and direct `python pipeline.py` from a checkout to
# find the package without first running `pip install -e .`.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deciphaer_image_segmentation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())