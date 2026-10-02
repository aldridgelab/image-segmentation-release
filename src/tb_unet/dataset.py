"""
Dataset module for TB U-Net training.

Handles loading source images + masks, creating 3-class labels,
and applying augmentations.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import tifffile
import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader

from tb_unet.augmentations import build_split_transform, build_transforms


class TBCellDataset(Dataset):
    """
    PyTorch Dataset for TB cell segmentation.

    Loads source images and creates 3-class labels:
    - Class 0: Background (pixels not covered by mask)
    - Class 1: Cell (central object with status='cell')
    - Class 2: Non-cell (central object with status='no_cell')

    Args:
        data_dir: Path to session_output directory containing masks/ and metadata/
        image_dir: Path to directory containing source TIFF stacks
        split: Which split to use ('train', 'val', 'test')
        channels: Which channels to use (default: [0] for Phase only)
        transform: Optional albumentations transform
        pad_multiple: Pad images to multiple of this value (for U-Net)
    """

    def __init__(
        self,
        data_dir: Path | str,
        image_dir: Path | str,
        split: Literal["train", "val", "test"] = "train",
        channels: list[int] | None = None,
        transform=None,
        pad_multiple: int = 32,
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir)
        self.split = split
        self.channels = channels if channels is not None else [0]  # Phase only
        self.transform = transform
        self.pad_multiple = pad_multiple
        self.debug_data = os.getenv("TB_UNET_DEBUG_DATA", "").lower() in {
            "1",
            "true",
            "yes",
        }

        self.masks_dir = self.data_dir / "masks"
        self.metadata_dir = self.data_dir / "metadata"
        self.image_index = self._build_image_index()

        # Load and filter samples
        self.samples = self._load_samples()
        logger.info(
            f"TBCellDataset initialized: {len(self.samples)} samples for {split}"
        )

    def _load_samples(self) -> pl.DataFrame:
        """Load sample metadata and filter to this split."""
        t0 = time.monotonic()
        records = []
        processed = 0
        missing_masks = 0
        missing_sources = 0

        for meta_path in self.metadata_dir.glob("*_meta.json"):
            processed += 1
            try:
                with open(meta_path) as f:
                    meta = json.load(f)

                status = meta.get("status", "")
                # Skip discarded items
                if status == "discard":
                    continue

                # Determine class label
                if status == "cell":
                    class_label = 1
                elif status == "no_cell":
                    class_label = 2
                else:
                    continue  # Skip unknown statuses

                image_id = meta.get("id", meta_path.stem.replace("_meta", ""))
                source_path = meta.get("source_path")

                # Build mask path
                mask_path = self.masks_dir / f"{image_id}_mask.tif"
                if not mask_path.exists():
                    missing_masks += 1
                    continue

                # Resolve source path using metadata or indexed image directory
                src_path = None
                if source_path:
                    source_candidate = Path(source_path)
                    if source_candidate.exists():
                        src_path = str(source_candidate)
                    else:
                        src_path = self.image_index.get(source_candidate.stem)
                if src_path is None:
                    src_path = self.image_index.get(image_id)
                if src_path is None:
                    missing_sources += 1
                    continue

                records.append({
                    "image_id": image_id,
                    "source_path": src_path,
                    "mask_path": str(mask_path),
                    "class_label": class_label,
                    "status": status,
                })
            except Exception as e:
                logger.warning(f"Error loading {meta_path}: {e}")
                continue
            if self.debug_data and processed % 1000 == 0:
                elapsed = time.monotonic() - t0
                logger.info(
                    "Metadata scan progress: {processed} files, "
                    "{records} usable, {missing_masks} missing masks, "
                    "{missing_sources} missing sources ({elapsed:.1f}s)",
                    processed=processed,
                    records=len(records),
                    missing_masks=missing_masks,
                    missing_sources=missing_sources,
                    elapsed=elapsed,
                )

        if not records:
            raise ValueError(f"No valid samples found in {self.data_dir}")

        df = pl.DataFrame(records)

        # Stratified split by class_label
        df = df.with_columns(
            pl.col("image_id").hash().mod(100).alias("hash_bucket")
        )

        # 70% train, 15% val, 15% test
        if self.split == "train":
            df = df.filter(pl.col("hash_bucket") < 70)
        elif self.split == "val":
            df = df.filter(
                (pl.col("hash_bucket") >= 70) & (pl.col("hash_bucket") < 85)
            )
        else:  # test
            df = df.filter(pl.col("hash_bucket") >= 85)

        if self.debug_data:
            elapsed = time.monotonic() - t0
            logger.info(
                "Metadata scan complete: {processed} files, {records} usable "
                "({elapsed:.1f}s)",
                processed=processed,
                records=len(records),
                elapsed=elapsed,
            )

        return df

    def _build_image_index(self) -> dict[str, str]:
        """Index image_dir once to avoid repeated globbing."""
        t0 = time.monotonic()
        index: dict[str, str] = {}
        if not self.image_dir.exists():
            logger.warning(f"Image directory does not exist: {self.image_dir}")
            return index
        for i, path in enumerate(self.image_dir.iterdir(), 1):
            if path.is_file():
                index[path.stem] = str(path)
            if self.debug_data and i % 2000 == 0:
                elapsed = time.monotonic() - t0
                logger.info(
                    "Image index progress: {count} files indexed ({elapsed:.1f}s)",
                    count=i,
                    elapsed=elapsed,
                )
        logger.info(
            f"Indexed {len(index)} images in {self.image_dir}"
        )
        return index

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_mask(self, mask: np.ndarray, *, image_id: str) -> np.ndarray:
        """
        Normalize mask to 2D (H, W).

        Accepts occasional singleton-channel masks (1, H, W) or (H, W, 1).
        """
        if mask.ndim == 2:
            return mask
        if mask.ndim == 3 and mask.shape[0] == 1:
            return mask[0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            return mask[..., 0]
        raise ValueError(
            f"{image_id}: unsupported mask shape {mask.shape}; expected (H,W)"
        )

    def _coerce_image_to_chw(
        self,
        image: np.ndarray,
        *,
        mask_shape: tuple[int, int],
        image_id: str,
        source_path: str,
    ) -> np.ndarray:
        """
        Coerce image arrays to CHW layout based on mask shape.

        Supports:
        - 2D grayscale: (H, W) -> (1, H, W)
        - CHW stacks: (C, H, W)
        - HWC stacks: (H, W, C) -> transpose to CHW
        """
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        elif image.ndim == 3:
            if image.shape[1:] == mask_shape:
                pass  # Already CHW
            elif image.shape[:2] == mask_shape and image.shape[-1] <= 16:
                # HWC stack; transpose to CHW.
                image = image.transpose(2, 0, 1)
                if self.debug_data:
                    logger.info(
                        "Converted HWC->CHW for image_id={} source={}",
                        image_id,
                        source_path,
                    )
            else:
                raise ValueError(
                    f"{image_id}: unsupported image shape {image.shape} for "
                    f"mask shape {mask_shape} (source={source_path})"
                )
        else:
            raise ValueError(
                f"{image_id}: unsupported image ndim={image.ndim} shape={image.shape} "
                f"(source={source_path})"
            )

        if image.shape[1:] != mask_shape:
            raise ValueError(
                f"{image_id}: image/mask shape mismatch after layout coercion: "
                f"image_hw={image.shape[1:]} mask_hw={mask_shape} "
                f"(source={source_path})"
            )
        return image

    def _select_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Select configured channels; zero-fill missing channels when needed.

        This mirrors inference behavior so mixed-width stacks do not crash
        dataset loading if a sample has fewer channels than requested.
        """
        if not self.channels:
            raise ValueError("channels must contain at least one channel index")

        max_requested = max(self.channels)
        if image.shape[0] <= max_requested:
            missing = (max_requested + 1) - image.shape[0]
            pad = np.zeros((missing, image.shape[1], image.shape[2]), dtype=image.dtype)
            image = np.concatenate([image, pad], axis=0)
            if self.debug_data:
                logger.info(
                    "Zero-filled {} missing channels (have={}, need={})",
                    missing,
                    image.shape[0] - missing,
                    max_requested + 1,
                )

        return image[self.channels]

    def __getitem__(self, idx: int) -> dict:
        row = self.samples.row(idx, named=True)

        # Load mask first; we use it to infer/validate image layout.
        mask = tifffile.imread(row["mask_path"])  # expected (H, W)
        mask = self._normalize_mask(mask, image_id=row["image_id"])

        # Load source image and coerce to CHW.
        image = tifffile.imread(row["source_path"])
        image = self._coerce_image_to_chw(
            image,
            mask_shape=tuple(mask.shape),
            image_id=row["image_id"],
            source_path=row["source_path"],
        )
        image = self._select_channels(image)

        # Normalize to [0, 1]
        image = image.astype(np.float32)
        for c in range(image.shape[0]):
            channel = image[c]
            cmin, cmax = channel.min(), channel.max()
            if cmax > cmin:
                image[c] = (channel - cmin) / (cmax - cmin)
            else:
                image[c] = 0.0

        # Create label map
        label = np.zeros_like(mask, dtype=np.int64)
        if mask.max() > 0:
            # Central object gets the class label (1 for cell, 2 for non-cell)
            label[mask > 0] = row["class_label"]
        # If mask is empty (rare no_cell with empty mask), label stays all 0 (background)

        # Apply transforms (expects HWC for albumentations)
        if self.transform is not None:
            # Transpose to HWC for albumentations
            image_hwc = image.transpose(1, 2, 0)
            transformed = self.transform(image=image_hwc, mask=label)
            image_hwc = transformed["image"]
            label = transformed["mask"]
            # Back to CHW
            image = image_hwc.transpose(2, 0, 1)

        # Pad to multiple of pad_multiple
        image, label, pad_info = self._pad_to_multiple(image, label)

        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label).long(),
            "image_id": row["image_id"],
            "class_label": row["class_label"],
            "pad_info": pad_info,
        }

    def _pad_to_multiple(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Pad image and label to multiple of pad_multiple."""
        _, h, w = image.shape
        new_h = ((h + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        new_w = ((w + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple

        pad_h = new_h - h
        pad_w = new_w - w

        if pad_h > 0 or pad_w > 0:
            # Pad image (reflect padding)
            image = np.pad(
                image,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="reflect",
            )
            # Pad label (constant 0 = background)
            label = np.pad(
                label,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        pad_info = {"original_h": h, "original_w": w, "pad_h": pad_h, "pad_w": pad_w}
        return image, label, pad_info


def get_train_transforms(augmentations_config: Any = None):
    """Get augmentation transforms for training."""
    return build_split_transform("train", augmentations_config)


def get_val_transforms(augmentations_config: Any = None):
    """Get transforms for validation."""
    return build_split_transform("val", augmentations_config)


def get_test_transforms(augmentations_config: Any = None):
    """Get transforms for test split."""
    return build_split_transform("test", augmentations_config)


def collate_variable_size_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function that pads each sample to the max H/W in the batch.

    Defined at module scope so it is pickleable with multiprocessing dataloaders
    on platforms that use spawn (for example macOS).
    """
    max_h = max(b["image"].shape[1] for b in batch)
    max_w = max(b["image"].shape[2] for b in batch)

    images = []
    labels = []
    for b in batch:
        img = b["image"]
        lbl = b["label"]

        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]

        if pad_h > 0 or pad_w > 0:
            # Pad image with 0 (already normalized)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
            # Pad label with 0 (background)
            lbl = torch.nn.functional.pad(lbl, (0, pad_w, 0, pad_h), mode="constant", value=0)

        images.append(img)
        labels.append(lbl)

    return {
        "image": torch.stack(images),
        "label": torch.stack(labels),
        "image_id": [b["image_id"] for b in batch],
        "class_label": [b["class_label"] for b in batch],
        "pad_info": [b["pad_info"] for b in batch],
    }


def create_dataloaders(
    data_dir: Path | str,
    image_dir: Path | str,
    batch_size: int = 16,
    num_workers: int = 4,
    channels: list[int] | None = None,
    pad_multiple: int = 32,
    augmentations_config: Any = None,
) -> dict[str, DataLoader]:
    """
    Create train, val, and test dataloaders.

    Args:
        data_dir: Path to session_output directory
        image_dir: Path to source images directory
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        channels: Which channels to use (default: [0] for Phase)
        pad_multiple: Pad images to multiple of this value
        augmentations_config: Augmentation strategy/probability config

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    transforms = build_transforms(augmentations_config)

    train_ds = TBCellDataset(
        data_dir=data_dir,
        image_dir=image_dir,
        split="train",
        channels=channels,
        transform=transforms["train"],
        pad_multiple=pad_multiple,
    )

    val_ds = TBCellDataset(
        data_dir=data_dir,
        image_dir=image_dir,
        split="val",
        channels=channels,
        transform=transforms["val"],
        pad_multiple=pad_multiple,
    )

    test_ds = TBCellDataset(
        data_dir=data_dir,
        image_dir=image_dir,
        split="test",
        channels=channels,
        transform=transforms["test"],
        pad_multiple=pad_multiple,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_variable_size_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_variable_size_batch,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_variable_size_batch,
    )

    logger.info(
        f"Created dataloaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
