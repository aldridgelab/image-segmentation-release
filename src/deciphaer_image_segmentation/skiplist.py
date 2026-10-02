from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


HEADER = """\
# Images listed here are excluded from MOMIA batches (no symlink is created).
# Lines are matched against the input filename (with or without extension) AND
# against the absolute source path. Comments start with '#'. Edit by hand or
# via `pipeline.py --config <yaml> --skip <name>`. Removing a line re-enables
# the image on the next run.
"""


def load(path: Path) -> set[str]:
    """Return the set of skip tokens (image stems / filenames / paths)."""
    if not path.exists():
        return set()
    tokens: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tokens.add(line)
    return tokens


def matches(skip_tokens: set[str], source: Path) -> bool:
    if not skip_tokens:
        return False
    candidates = {source.name, source.stem, str(source.resolve())}
    return bool(candidates & skip_tokens)


def append(path: Path, tokens: list[str], *, comment: str = "") -> list[str]:
    """Append `tokens` to skip.txt under a dated comment, skipping duplicates.

    Returns the tokens that were actually appended (i.e. not already present).
    """
    existing = load(path)
    fresh = [t for t in tokens if t and t not in existing]
    if not fresh:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ""
    if not path.exists():
        body += HEADER
    body += "\n# "
    body += datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if comment:
        body += f" — {comment}"
    body += "\n"
    body += "\n".join(fresh) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(body)
    return fresh
