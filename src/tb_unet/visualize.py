"""
Visualization utilities for TB U-Net.

Provides functions for visualizing predictions, training progress,
and model analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger


# Color map for 3-class segmentation
CLASS_COLORS = {
    0: [0, 0, 0],        # Background: black
    1: [0, 255, 0],      # Cell: green
    2: [255, 0, 0],      # Non-cell: red
}


def create_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create an overlay of prediction on image.

    Args:
        image: Grayscale image (H, W) normalized to [0, 1]
        prediction: Class predictions (H, W)
        alpha: Overlay transparency

    Returns:
        RGB overlay image (H, W, 3)
    """
    # Convert grayscale to RGB
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image.transpose(1, 2, 0)

    # Scale to [0, 255]
    rgb = (rgb * 255).astype(np.uint8)

    # Create color mask
    color_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[prediction == class_id] = color

    # Blend
    overlay = (alpha * color_mask + (1 - alpha) * rgb).astype(np.uint8)

    return overlay


def visualize_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    n_samples: int = 4,
    figsize: tuple[int, int] = (16, 4),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Visualize a batch of predictions.

    Args:
        images: Input images (B, C, H, W)
        labels: Ground truth labels (B, H, W)
        predictions: Model predictions (B, H, W)
        n_samples: Number of samples to show
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    n_samples = min(n_samples, images.shape[0])

    fig, axes = plt.subplots(n_samples, 4, figsize=(figsize[0], figsize[1] * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        img = images[i, 0].cpu().numpy()  # Take first channel
        lbl = labels[i].cpu().numpy()
        pred = predictions[i].cpu().numpy()

        # Normalize image for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Image
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        # Ground truth
        gt_overlay = create_overlay(img, lbl)
        axes[i, 1].imshow(gt_overlay)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        # Prediction
        pred_overlay = create_overlay(img, pred)
        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

        # Difference (errors)
        diff = (lbl != pred).astype(np.float32)
        axes[i, 3].imshow(img, cmap="gray")
        axes[i, 3].imshow(diff, cmap="Reds", alpha=0.5)
        axes[i, 3].set_title("Errors")
        axes[i, 3].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {save_path}")

    return fig


def visualize_sample_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 8,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """
    Generate visualization of model predictions on samples from dataloader.

    Args:
        model: Trained model
        dataloader: Data loader to sample from
        device: Device to run inference on
        n_samples: Number of samples to visualize
        save_dir: Optional directory to save figures

    Returns:
        Matplotlib figure
    """
    model.eval()

    # Get batch
    batch = next(iter(dataloader))
    images = batch["image"][:n_samples].to(device)
    labels = batch["label"][:n_samples]

    with torch.no_grad():
        logits = model(images)
        predictions = torch.argmax(logits, dim=1).cpu()

    fig = visualize_batch(images.cpu(), labels, predictions, n_samples=n_samples)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "sample_predictions.png", dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        confusion_matrix: (num_classes, num_classes) array
        class_names: List of class names
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ["Background", "Cell", "Non-cell"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize by row (true class)
    cm_normalized = confusion_matrix.astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm_normalized, row_sums, where=row_sums != 0)

    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm_normalized[i, j]:.2f}\n({confusion_matrix[i, j]:,})"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")

    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_metrics: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot training curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_metrics: Dict of metric name -> list of values per epoch
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    n_plots = 1 + len(val_metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    axes[0].plot(epochs, train_losses, label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, val_losses, label="Val", marker="s", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metric curves
    for i, (metric_name, values) in enumerate(val_metrics.items()):
        ax = axes[i + 1]
        ax.plot(epochs, values, marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Validation {metric_name}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved training curves to {save_path}")

    return fig
