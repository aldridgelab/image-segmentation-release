"""
Evaluation metrics for TB U-Net.

Includes pixel-level metrics (Dice, IoU) and object-level metrics
(precision, recall, F1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import ndimage
from loguru import logger


@dataclass
class SegmentationMetrics:
    """Container for segmentation metrics."""

    # Pixel-level metrics per class
    dice_per_class: dict[int, float] = field(default_factory=dict)
    iou_per_class: dict[int, float] = field(default_factory=dict)

    # Mean metrics
    dice_mean: float = 0.0
    iou_mean: float = 0.0

    # Object-level metrics for cell class
    cell_precision: float = 0.0
    cell_recall: float = 0.0
    cell_f1: float = 0.0

    # Non-cell rejection metrics
    non_cell_true_negative_rate: float = 0.0  # Correctly rejected non-cells
    non_cell_false_positive_rate: float = 0.0  # Non-cells incorrectly called cells

    # Counts
    n_samples: int = 0
    confusion_matrix: np.ndarray | None = None


def compute_dice(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """Compute Dice coefficient for a single class."""
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)

    intersection = np.sum(pred_mask & target_mask)
    union = np.sum(pred_mask) + np.sum(target_mask)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / union


def compute_iou(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """Compute IoU (Jaccard index) for a single class."""
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)

    intersection = np.sum(pred_mask & target_mask)
    union = np.sum(pred_mask | target_mask)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def compute_confusion_matrix(
    pred: np.ndarray, target: np.ndarray, num_classes: int = 3
) -> np.ndarray:
    """Compute confusion matrix for pixel-level predictions."""
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            conf_matrix[true_class, pred_class] = np.sum(
                (target == true_class) & (pred == pred_class)
            )

    return conf_matrix


class MetricsCalculator:
    """
    Calculator for accumulating and computing segmentation metrics.

    Tracks metrics across batches and computes final statistics.
    """

    CLASS_NAMES = {0: "background", 1: "cell", 2: "non_cell"}

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self.dice_scores = {i: [] for i in range(self.num_classes)}
        self.iou_scores = {i: [] for i in range(self.num_classes)}
        self.n_samples = 0

        # Object-level tracking
        self.object_predictions = []  # List of (true_class, pred_class) per object

    def update(
        self,
        preds: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        class_labels: list[int] | None = None,
    ):
        """
        Update metrics with a batch of predictions.

        Args:
            preds: (B, H, W) predicted class indices
            targets: (B, H, W) ground truth class indices
            class_labels: Optional list of true object classes (for object-level metrics)
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        batch_size = preds.shape[0]

        for i in range(batch_size):
            pred = preds[i]
            target = targets[i]

            # Update confusion matrix
            self.confusion_matrix += compute_confusion_matrix(
                pred, target, self.num_classes
            )

            # Update per-class Dice and IoU
            for class_id in range(self.num_classes):
                self.dice_scores[class_id].append(compute_dice(pred, target, class_id))
                self.iou_scores[class_id].append(compute_iou(pred, target, class_id))

            # Object-level metrics (per-image)
            if class_labels is not None:
                true_class = class_labels[i]
                # Determine predicted object class (majority vote in central region)
                pred_class = self._get_predicted_object_class(pred, target)
                self.object_predictions.append((true_class, pred_class))

            self.n_samples += 1

    def _get_predicted_object_class(
        self, pred: np.ndarray, target: np.ndarray
    ) -> int:
        """
        Determine the predicted class for the central object.

        Uses majority vote over pixels where target is non-background.
        """
        # Get pixels where target has an object (class 1 or 2)
        object_mask = target > 0

        if not np.any(object_mask):
            return 0  # No object in target, return background

        # Get predicted classes for those pixels
        pred_in_object = pred[object_mask]

        # Majority vote (excluding background)
        cell_count = np.sum(pred_in_object == 1)
        non_cell_count = np.sum(pred_in_object == 2)
        bg_count = np.sum(pred_in_object == 0)

        # If majority is background, check if more cell or non-cell
        if bg_count > cell_count + non_cell_count:
            # Model thinks there's no object
            return 0

        if cell_count >= non_cell_count:
            return 1
        return 2

    def compute(self) -> SegmentationMetrics:
        """Compute final metrics from accumulated data."""
        metrics = SegmentationMetrics(n_samples=self.n_samples)

        # Pixel-level metrics
        for class_id in range(self.num_classes):
            if self.dice_scores[class_id]:
                metrics.dice_per_class[class_id] = np.mean(self.dice_scores[class_id])
                metrics.iou_per_class[class_id] = np.mean(self.iou_scores[class_id])

        # Mean metrics (excluding background)
        fg_dice = [
            metrics.dice_per_class[i]
            for i in [1, 2]
            if i in metrics.dice_per_class
        ]
        fg_iou = [
            metrics.iou_per_class[i]
            for i in [1, 2]
            if i in metrics.iou_per_class
        ]

        if fg_dice:
            metrics.dice_mean = np.mean(fg_dice)
        if fg_iou:
            metrics.iou_mean = np.mean(fg_iou)

        # Object-level metrics
        if self.object_predictions:
            self._compute_object_metrics(metrics)

        metrics.confusion_matrix = self.confusion_matrix

        return metrics

    def _compute_object_metrics(self, metrics: SegmentationMetrics):
        """Compute object-level precision/recall/F1."""
        # Object-level confusion matrix
        # True class vs predicted class for each image's central object
        obj_conf = np.zeros((3, 3), dtype=np.int64)
        for true_class, pred_class in self.object_predictions:
            obj_conf[true_class, pred_class] += 1

        # Cell precision: TP / (TP + FP)
        # TP = predicted cell, actually cell
        # FP = predicted cell, actually non-cell
        tp_cell = obj_conf[1, 1]  # True cell, pred cell
        fp_cell = obj_conf[2, 1]  # True non-cell, pred cell
        fn_cell = obj_conf[1, 2] + obj_conf[1, 0]  # True cell, pred non-cell or bg

        if tp_cell + fp_cell > 0:
            metrics.cell_precision = tp_cell / (tp_cell + fp_cell)
        if tp_cell + fn_cell > 0:
            metrics.cell_recall = tp_cell / (tp_cell + fn_cell)
        if metrics.cell_precision + metrics.cell_recall > 0:
            metrics.cell_f1 = (
                2 * metrics.cell_precision * metrics.cell_recall
                / (metrics.cell_precision + metrics.cell_recall)
            )

        # Non-cell rejection metrics
        # True negative: non-cell correctly identified as non-cell or background
        tn_non_cell = obj_conf[2, 2] + obj_conf[2, 0]
        total_non_cell = obj_conf[2, :].sum()

        if total_non_cell > 0:
            metrics.non_cell_true_negative_rate = tn_non_cell / total_non_cell
            metrics.non_cell_false_positive_rate = obj_conf[2, 1] / total_non_cell

    def summary(self) -> dict:
        """Return metrics as a flat dictionary for logging."""
        m = self.compute()

        result = {
            "n_samples": m.n_samples,
            "dice_mean": m.dice_mean,
            "iou_mean": m.iou_mean,
            "cell_precision": m.cell_precision,
            "cell_recall": m.cell_recall,
            "cell_f1": m.cell_f1,
            "non_cell_tnr": m.non_cell_true_negative_rate,
            "non_cell_fpr": m.non_cell_false_positive_rate,
        }

        # Add per-class metrics
        for class_id, name in self.CLASS_NAMES.items():
            if class_id in m.dice_per_class:
                result[f"dice_{name}"] = m.dice_per_class[class_id]
            if class_id in m.iou_per_class:
                result[f"iou_{name}"] = m.iou_per_class[class_id]

        return result


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> SegmentationMetrics:
    """
    Evaluate model on a dataloader.

    Args:
        model: Trained U-Net model
        dataloader: Validation or test dataloader
        device: Device to run evaluation on

    Returns:
        SegmentationMetrics with all computed metrics
    """
    model.eval()
    calculator = MetricsCalculator()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            class_labels = batch["class_label"]

            # Forward pass
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            # Update metrics
            calculator.update(preds, labels, class_labels)

    return calculator.compute()


def print_metrics(metrics: SegmentationMetrics, title: str = "Evaluation Metrics"):
    """Pretty-print metrics to console."""
    logger.info(f"\n{'=' * 50}")
    logger.info(f"{title}")
    logger.info(f"{'=' * 50}")
    logger.info(f"Samples: {metrics.n_samples}")
    logger.info("")
    logger.info("Pixel-level metrics:")
    logger.info(f"  Mean Dice (fg): {metrics.dice_mean:.4f}")
    logger.info(f"  Mean IoU (fg):  {metrics.iou_mean:.4f}")
    for class_id, name in MetricsCalculator.CLASS_NAMES.items():
        if class_id in metrics.dice_per_class:
            logger.info(
                f"  {name:12} - Dice: {metrics.dice_per_class[class_id]:.4f}, "
                f"IoU: {metrics.iou_per_class[class_id]:.4f}"
            )
    logger.info("")
    logger.info("Object-level metrics (cell detection):")
    logger.info(f"  Precision: {metrics.cell_precision:.4f}")
    logger.info(f"  Recall:    {metrics.cell_recall:.4f}")
    logger.info(f"  F1 Score:  {metrics.cell_f1:.4f}")
    logger.info("")
    logger.info("Non-cell rejection:")
    logger.info(f"  True Negative Rate: {metrics.non_cell_true_negative_rate:.4f}")
    logger.info(f"  False Positive Rate: {metrics.non_cell_false_positive_rate:.4f}")
    logger.info(f"{'=' * 50}\n")
