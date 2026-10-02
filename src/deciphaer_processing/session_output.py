"""
Helpers for exporting measurements from reviewed ``session_output`` runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


STATUS_ALIASES: dict[str, str] = {
    "pass": "cell",
    "fail": "no_cell",
    "discarded": "discard",
}

CHANNEL_ALIASES: dict[str, str] = {
    "phase": "Phase",
    "af405": "AF405",
    "hada": "AF405",
    "bod493": "Bod493",
    "bodipy": "Bod493",
}

DEFAULT_CHANNEL_NAMES: tuple[str, ...] = ("Phase", "AF405", "Bod493")
TIFF_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff", ".TIF", ".TIFF")


def normalize_review_status(status: str | None) -> str:
    """Normalize legacy review statuses to the current terminology."""
    normalized = str(status or "").strip().lower()
    return STATUS_ALIASES.get(normalized, normalized)


def canonicalize_channel_names(
    names: Sequence[str] | None,
    channel_count: int,
) -> list[str]:
    """
    Normalize channel aliases to the feature-export naming convention.

    The processing pipeline expects canonical fluorescence names like
    ``AF405`` and ``Bod493`` even if TIFF metadata uses ``HADA`` or
    ``Bodipy``.
    """
    output: list[str] = []
    for idx in range(max(0, channel_count)):
        if names is not None and idx < len(names):
            raw = str(names[idx]).strip()
        elif idx < len(DEFAULT_CHANNEL_NAMES):
            raw = DEFAULT_CHANNEL_NAMES[idx]
        else:
            raw = f"Channel {idx + 1}"

        canonical = CHANNEL_ALIASES.get(raw.lower(), raw or f"Channel {idx + 1}")
        output.append(canonical)
    return output


def resolve_channel_names(image_path: Path, channel_count: int) -> list[str]:
    """Resolve channel names from TIFF metadata, falling back to defaults."""
    detected: Sequence[str] | None = None
    try:
        from app.backend.channel_utils import suggest_channel_names_from_tiff

        detected, _ = suggest_channel_names_from_tiff(image_path, channel_count)
    except Exception:
        detected = None

    return canonicalize_channel_names(detected, channel_count)


def load_annotation_maps(annotation_path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    """Load per-cell notes and include/exclude overrides from an annotations JSON."""
    if annotation_path is None or not annotation_path.exists():
        return {}, {}

    data = json.loads(annotation_path.read_text())

    note_map: dict[str, str] = {}
    for key, value in dict(data.get("cell_notes", {})).items():
        clean = str(value).strip()
        if clean:
            note_map[str(key)] = clean

    override_map: dict[str, str] = {}
    for key, value in dict(data.get("cell_overrides", {})).items():
        clean = str(value).strip().lower()
        if clean in {"include", "exclude"}:
            override_map[str(key)] = clean

    return note_map, override_map


def resolve_central_label(
    metadata: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]] | None = None,
) -> int | None:
    """Resolve the saved central-cell label from session metadata."""
    centrality = metadata.get("centrality")
    if isinstance(centrality, Mapping):
        label = centrality.get("central_label")
        try:
            if label is not None:
                return int(label)
        except (TypeError, ValueError):
            pass

    for cell in cells or ():
        if bool(cell.get("is_central", False)):
            try:
                return int(cell.get("label"))
            except (TypeError, ValueError):
                return None

    return None


def apply_cell_annotations(
    cell: Mapping[str, Any],
    *,
    note_map: Mapping[str, str] | None = None,
    override_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a cell record with annotation overrides merged in."""
    note_map = note_map or {}
    override_map = override_map or {}

    effective = dict(cell)
    label_key = str(effective.get("label", ""))

    override = override_map.get(label_key, effective.get("override"))
    if override not in {"include", "exclude"}:
        override = None

    included = bool(effective.get("included", False))
    if override == "include":
        included = True
    elif override == "exclude":
        included = False

    note = note_map.get(label_key, effective.get("note"))
    if note is not None and str(note).strip() == "":
        note = None

    effective["included"] = included
    effective["override"] = override
    effective["note"] = note
    return effective


def resolve_image_path(
    metadata: Mapping[str, Any],
    image_id: str,
    *,
    images_dir: Path | None = None,
) -> Path:
    """Resolve a source image path from session metadata plus optional fallbacks."""
    candidates: list[Path] = []

    source_path_value = metadata.get("source_path")
    if source_path_value:
        source_path = Path(str(source_path_value)).expanduser()
        candidates.append(source_path)
        if images_dir is not None:
            candidates.append(images_dir / source_path.name)

    if images_dir is not None:
        for ext in TIFF_EXTENSIONS:
            candidates.append(images_dir / f"{image_id}{ext}")

    seen: set[Path] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
        if candidate.exists():
            return candidate

    attempted = ", ".join(str(path) for path in deduped) or "<none>"
    raise FileNotFoundError(
        f"Could not resolve source image for {image_id}. Checked: {attempted}"
    )


def resolve_mask_path(masks_dir: Path, image_id: str) -> Path:
    """Resolve a labeled mask TIFF for a session-output item."""
    candidates = [masks_dir / f"{image_id}_mask{ext}" for ext in TIFF_EXTENSIONS]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve mask for {image_id}. Checked: {attempted}")
