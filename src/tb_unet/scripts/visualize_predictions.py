#!/usr/bin/env python
"""
Quick validation visualization script.

Shows side-by-side comparisons of:
- Raw image
- P(Cell)
- P(Non Cell)
- P(Background)

Usage:
    python tb_unet/scripts/visualize_predictions.py --checkpoint tb_unet/checkpoints/phase_only_v1/best.pt
    python tb_unet/scripts/visualize_predictions.py --checkpoint tb_unet/checkpoints/sanity_test/best.pt --n 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tb_unet.dataset import TBCellDataset, get_val_transforms
from tb_unet.inference import load_model


def visualize_predictions(
    checkpoint_path: str,
    data_dir: str = "session_output",
    image_dir: str = "data/images",
    split: str = "val",
    n_samples: int = 8,
    n_cells: int | None = None,
    n_non_cells: int | None = None,
    save_path: str | None = None,
    device: str | None = None,
):
    """
    Visualize model predictions on validation/test samples.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to session_output directory
        image_dir: Path to source images
        split: Which split to use ('val' or 'test')
        n_samples: Total number of samples to show (if n_cells/n_non_cells not specified)
        n_cells: Number of cell samples to show
        n_non_cells: Number of non-cell samples to show
        save_path: Optional path to save figure
        device: Device to run inference on
    """
    # Load model
    model, config = load_model(checkpoint_path, device=device)
    device = next(model.parameters()).device

    # Load dataset
    channels = config.get("data", {}).get("channels", config.get("channels", [0]))
    dataset = TBCellDataset(
        data_dir=data_dir,
        image_dir=image_dir,
        split=split,
        channels=channels,
        transform=None,  # No augmentation for visualization
    )

    logger.info(f"Loaded {len(dataset)} samples from {split} split")

    # Select samples
    if n_cells is not None or n_non_cells is not None:
        # Sample specific numbers of each class
        n_cells = n_cells or 0
        n_non_cells = n_non_cells or 0

        cell_indices = [i for i in range(len(dataset)) if dataset.samples.row(i, named=True)["class_label"] == 1]
        non_cell_indices = [i for i in range(len(dataset)) if dataset.samples.row(i, named=True)["class_label"] == 2]

        np.random.shuffle(cell_indices)
        np.random.shuffle(non_cell_indices)

        indices = cell_indices[:n_cells] + non_cell_indices[:n_non_cells]
    else:
        # Random sample
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    n_show = len(indices)

    # Create figure
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    model.eval()

    for i, idx in enumerate(indices):
        sample = dataset[idx]

        image = sample["image"]  # (C, H, W)
        label = sample["label"]  # (H, W)
        class_label = sample["class_label"]

        # Get original size before padding
        pad_info = sample["pad_info"]
        orig_h, orig_w = pad_info["original_h"], pad_info["original_w"]

        # Run prediction
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Crop back to original size
        image_np = image[0, :orig_h, :orig_w].numpy()  # First channel
        label_np = label[:orig_h, :orig_w].numpy()
        probs_np = probs[:, :orig_h, :orig_w]

        # Binary labeling (cell vs non_cell)
        true_binary = "cell" if class_label == 1 else "non_cell"
        true_bit = 1 if class_label == 1 else 0

        region_mask = label_np > 0
        if not np.any(region_mask):
            region_mask = np.ones_like(label_np, dtype=bool)

        cell_score = float(probs_np[1][region_mask].mean())
        non_cell_score = float(probs_np[2][region_mask].mean())
        pred_binary = "cell" if cell_score >= non_cell_score else "non_cell"
        pred_bit = 1 if pred_binary == "cell" else 0
        correct = "✓" if pred_binary == true_binary else "✗"

        # Plot
        # 1. Raw image
        axes[i, 0].imshow(image_np, cmap="gray")
        axes[i, 0].set_title(
            f"Raw\nTrue: {true_binary} ({true_bit}) | Pred: {pred_binary} ({pred_bit}) {correct}"
        )
        axes[i, 0].axis("off")

        # 2. P(Cell)
        axes[i, 1].imshow(probs_np[1], cmap="magma", vmin=0, vmax=1)
        axes[i, 1].set_title("P(Cell)")
        axes[i, 1].axis("off")

        # 3. P(Non Cell)
        axes[i, 2].imshow(probs_np[2], cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title("P(Non Cell)")
        axes[i, 2].axis("off")

        # 4. P(Background)
        axes[i, 3].imshow(probs_np[0], cmap="magma", vmin=0, vmax=1)
        axes[i, 3].set_title("P(Background)")
        axes[i, 3].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {save_path}")

    plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize TB U-Net predictions")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="session_output",
        help="Path to session_output directory",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/images",
        help="Path to source images",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to visualize",
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=8,
        help="Number of samples to show",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=None,
        help="Number of cell samples to show",
    )
    parser.add_argument(
        "--n-non-cells",
        type=int,
        default=None,
        help="Number of non-cell samples to show",
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Path to save figure",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cpu, cuda, mps)",
    )

    args = parser.parse_args()

    visualize_predictions(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        split=args.split,
        n_samples=args.n,
        n_cells=args.n_cells,
        n_non_cells=args.n_non_cells,
        save_path=args.save,
        device=args.device,
    )


if __name__ == "__main__":
    main()
