"""
Deterministic cell hashing for traceability.

Generates unique, reproducible cell IDs based on source information.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def generate_cell_hash(
    source_path: str | Path,
    mask_label: int,
    bbox: tuple[int, int, int, int],
    hash_length: int = 16,
) -> str:
    """
    Generate a deterministic SHA256 hash for a cell.

    The hash is based on:
    - Source image path (filename only, for portability)
    - Mask label (integer label in the mask)
    - Bounding box (min_row, min_col, max_row, max_col)

    This ensures the same cell always gets the same hash, regardless of
    when or where the processing is run.

    Args:
        source_path: Path to source image file
        mask_label: Integer label of the cell in the mask
        bbox: Bounding box tuple (min_row, min_col, max_row, max_col)
        hash_length: Number of hex characters to return (default 16)

    Returns:
        Hexadecimal hash string (e.g., "a1b2c3d4e5f6g7h8")
    """
    # Use filename only for portability across systems
    if isinstance(source_path, Path):
        filename = source_path.name
    else:
        filename = Path(source_path).name

    # Create deterministic string representation
    hash_input = f"{filename}|{mask_label}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    # SHA256 hash
    hash_obj = hashlib.sha256(hash_input.encode("utf-8"))
    full_hash = hash_obj.hexdigest()

    return full_hash[:hash_length]


def generate_cell_id(
    source_path: str | Path,
    mask_label: int,
    bbox: tuple[int, int, int, int],
    prefix: str = "sc_",
) -> str:
    """
    Generate a formatted cell ID with prefix.

    Args:
        source_path: Path to source image file
        mask_label: Integer label of the cell in the mask
        bbox: Bounding box tuple (min_row, min_col, max_row, max_col)
        prefix: Prefix for the cell ID (default "sc_" for single-cell)

    Returns:
        Cell ID string (e.g., "sc_a1b2c3d4e5f6g7h8")
    """
    cell_hash = generate_cell_hash(source_path, mask_label, bbox)
    return f"{prefix}{cell_hash}"


def parse_filename_metadata(filename: str) -> dict[str, str]:
    """
    Parse metadata fields from standardized filename.

    Expected format:
    stack_{N}-tf_HADA_BOD_d1_bridge_060225_{PLATE}_s{SCENE}_m{POSITION}_Phase

    Args:
        filename: Image filename (with or without extension)

    Returns:
        Dictionary with plate, scene, position fields
    """
    # Remove extension if present
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]

    metadata = {
        "plate": "UNK",
        "scene": "UNK",
        "position": "UNK",
    }

    try:
        # Split by underscores
        parts = filename.split("_")

        # Find plate (4-digit number after date)
        for i, part in enumerate(parts):
            # Look for date pattern (6 digits like 060225)
            if len(part) == 6 and part.isdigit():
                # Next part should be plate
                if i + 1 < len(parts):
                    plate_part = parts[i + 1]
                    if plate_part.isdigit() and len(plate_part) == 4:
                        metadata["plate"] = plate_part
                break

        # Find scene (sXX pattern)
        for part in parts:
            if part.startswith("s") and len(part) >= 2:
                scene_num = part[1:]
                if scene_num.isdigit():
                    metadata["scene"] = scene_num
                    break

        # Find position (mXX pattern)
        for part in parts:
            if part.startswith("m") and len(part) >= 2:
                pos_num = part[1:]
                if pos_num.isdigit():
                    metadata["position"] = pos_num
                    break

    except Exception:
        pass  # Return defaults on any parsing error

    return metadata


def hash_from_metadata(
    image_id: str,
    label: int,
    centroid: tuple[float, float],
) -> str:
    """
    Alternative hash generation using metadata fields.

    Useful when bbox is not available but centroid is.

    Args:
        image_id: Image identifier string
        label: Cell label
        centroid: Cell centroid (row, col)

    Returns:
        Hexadecimal hash string
    """
    # Round centroid to integer for reproducibility
    centroid_int = (int(round(centroid[0])), int(round(centroid[1])))
    hash_input = f"{image_id}|{label}|{centroid_int[0]},{centroid_int[1]}"

    hash_obj = hashlib.sha256(hash_input.encode("utf-8"))
    return hash_obj.hexdigest()[:16]
