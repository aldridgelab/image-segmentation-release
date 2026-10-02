"""
Loss functions for TB U-Net training.

Includes weighted cross-entropy, Dice loss, and combined losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for multi-class segmentation.

    Computes per-class Dice coefficient and returns 1 - mean(Dice).
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -100,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) class indices

        Returns:
            Scalar Dice loss
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Create mask for ignored pixels
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        # Compute per-class Dice
        dims = (0, 2, 3)  # Sum over batch and spatial dims
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Weight classes
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=dice.device)
            dice = dice * weights
            dice_loss = 1.0 - dice.sum() / weights.sum()
        else:
            dice_loss = 1.0 - dice.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: list[float] | None = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) class indices

        Returns:
            Scalar focal loss
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )

        # Get probabilities for true class
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Class weights
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=logits.device)[targets]
            focal_weight = focal_weight * alpha_t

        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float()
            focal_loss = (focal_weight * ce_loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            focal_loss = (focal_weight * ce_loss).mean()

        return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted cross-entropy + Dice loss.

    This is the recommended loss for the TB U-Net model.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: list[float] | None = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        # Cross-entropy with class weights
        if class_weights is not None:
            weight = torch.tensor(class_weights)
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Dice loss
        self.dice = DiceLoss(ignore_index=ignore_index, class_weights=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) class indices

        Returns:
            Dictionary with 'loss', 'ce_loss', and 'dice_loss'
        """
        # Move CE weights to same device as logits
        if self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(logits.device)

        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "dice_loss": dice_loss,
        }


class CombinedFocalDiceLoss(nn.Module):
    """
    Combined loss: Focal loss + Dice loss.

    Use this if you have severe class imbalance and
    standard CE+Dice isn't handling non-cells well.
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        alpha: list[float] | None = None,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index, class_weights=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss

        return {
            "loss": total_loss,
            "focal_loss": focal_loss,
            "dice_loss": dice_loss,
        }


def compute_class_weights(
    class_counts: list[int],
    method: str = "inverse_freq",
    smooth: float = 1.0,
) -> list[float]:
    """
    Compute class weights from class counts.

    Args:
        class_counts: List of pixel counts per class [bg, cell, non_cell]
        method: 'inverse_freq', 'inverse_sqrt_freq', or 'effective_num'
        smooth: Smoothing factor for effective_num method

    Returns:
        List of class weights
    """
    import numpy as np

    counts = np.array(class_counts, dtype=float)
    total = counts.sum()

    if method == "inverse_freq":
        # Inverse frequency weighting
        weights = total / (len(counts) * counts + 1e-6)
    elif method == "inverse_sqrt_freq":
        # Square root inverse frequency (less aggressive)
        weights = np.sqrt(total / (len(counts) * counts + 1e-6))
    elif method == "effective_num":
        # Effective number of samples (from Class-Balanced Loss paper)
        beta = (total - 1) / total
        effective_num = (1 - np.power(beta, counts)) / (1 - beta)
        weights = 1 / (effective_num + smooth)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Normalize so weights sum to num_classes
    weights = weights / weights.sum() * len(weights)

    return weights.tolist()
