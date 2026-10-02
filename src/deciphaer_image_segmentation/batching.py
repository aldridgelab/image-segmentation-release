from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchRecord:
    batch: str
    source_path: str
    link_path: str


def discover_images(image_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = image_dir.rglob(pattern) if recursive else image_dir.glob(pattern)
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def setup_batch_folders(
    *,
    image_dir: Path,
    batch_dir: Path,
    n_batches: int,
    pattern: str,
    recursive: bool,
    overwrite_links: bool,
    skip_tokens: set[str] | None = None,
) -> list[BatchRecord]:
    """Distribute images across batch dirs as symlinks.

    `skip_tokens` is a set of names matched against each image's filename, stem,
    or absolute path; matches are skipped (no symlink). Iteration still walks the
    full image list so the surviving images keep their batch assignment — adding
    or removing a skip entry doesn't reshuffle every batch.
    """
    from deciphaer_image_segmentation import skiplist

    if n_batches < 1:
        raise ValueError("n_batches must be at least 1")
    images = discover_images(image_dir=image_dir, pattern=pattern, recursive=recursive)
    if not images:
        raise FileNotFoundError(f"No TIFF images matching {pattern!r} under {image_dir}")

    batch_dir.mkdir(parents=True, exist_ok=True)
    skip_tokens = skip_tokens or set()

    # Fast-path: if the manifest from a prior setup already covers the exact
    # same eligible image set we'd link this time, return the cached records
    # and skip the per-file symlink loop. On a 5k-image input that's the
    # difference between ~3s of stat+symlink ops and a single CSV read.
    cached = _try_read_cached_manifest(
        batch_dir=batch_dir,
        images=images,
        skip_tokens=skip_tokens,
    )
    if cached is not None:
        _log.info(
            "setup: manifest cache hit (%d images unchanged) — skipping relink",
            len(cached),
        )
        return cached

    records: list[BatchRecord] = []
    seen_targets: set[Path] = set()
    for idx, source in enumerate(images):
        batch = f"batch{(idx % n_batches) + 1}"
        target_dir = batch_dir / batch
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if target in seen_targets:
            digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:8]
            target = target_dir / f"{source.stem}_{digest}{source.suffix}"
        seen_targets.add(target)
        if skiplist.matches(skip_tokens, source):
            if target.is_symlink() or target.exists():
                target.unlink()
            continue
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                records.append(BatchRecord(batch=batch, source_path=str(source.resolve()), link_path=str(target)))
                continue
            if not overwrite_links:
                raise FileExistsError(f"Batch target already exists: {target}")
            target.unlink()
        target.symlink_to(source.resolve())
        records.append(BatchRecord(batch=batch, source_path=str(source.resolve()), link_path=str(target)))

    write_batch_manifest(batch_dir / "batch_manifest.csv", records)
    return records


def _try_read_cached_manifest(
    *,
    batch_dir: Path,
    images: list[Path],
    skip_tokens: set[str],
) -> list[BatchRecord] | None:
    """Return the cached manifest if the eligible image set is unchanged.

    Returns ``None`` (forcing a full re-link) when:
      - manifest is absent / unreadable,
      - the set of source paths the manifest covers differs from what we'd
        link now (after applying ``skip_tokens``),
      - any manifest target is no longer a symlink (so a stage that needs
        the link would otherwise fail mid-run).

    Resolution comparison is by absolute source path. We sample a few
    target symlinks rather than stat'ing all 5k+ to keep the fast path fast;
    any drift will be caught when the relevant stage actually opens the
    target. The win is removing the per-file symlink dance on resubmits.
    """
    from deciphaer_image_segmentation import skiplist

    manifest_path = batch_dir / "batch_manifest.csv"
    if not manifest_path.is_file():
        return None
    try:
        cached = read_batch_manifest(manifest_path)
    except (OSError, csv.Error, KeyError, ValueError):
        return None
    if not cached:
        return None

    eligible_sources = {
        str(img.resolve())
        for img in images
        if not skiplist.matches(skip_tokens, img)
    }
    cached_sources = {Path(r.source_path).as_posix() for r in cached}
    # source_path entries were already resolved at write time, so a string
    # compare is sufficient and avoids 5k+ stat calls.
    if cached_sources != eligible_sources:
        return None

    # Spot-check up to 8 targets — if the user blew away batch_dir between
    # runs we want to fall back to the full path.
    sample = cached[:: max(1, len(cached) // 8)] if len(cached) > 8 else cached
    for record in sample:
        link = Path(record.link_path)
        if not link.is_symlink():
            return None

    return cached


def write_batch_manifest(path: Path, records: list[BatchRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch", "source_path", "link_path"])
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def read_batch_manifest(path: Path) -> list[BatchRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [BatchRecord(**row) for row in csv.DictReader(handle)]
