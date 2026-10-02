"""
Inference module for TB U-Net.

Provides functions for loading trained models and running predictions,
designed for integration with the main segmentation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tifffile
from loguru import logger

from tb_unet.model import create_model


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Load a trained TB U-Net model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load model on (auto-detects if None)

    Returns:
        Tuple of (model, config_dict)
    """
    checkpoint_path = Path(checkpoint_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    logger.info(f"Loading model from {checkpoint_path} on {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model_kwargs = _model_kwargs_from_config(config)
    model = create_model(**model_kwargs)

    state_dict = _normalize_checkpoint_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully (epoch {checkpoint.get('epoch', '?')})")

    return model, config


def _lookup(config: dict, *path: str, default=None):
    node = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _model_kwargs_from_config(config: dict) -> dict:
    # Nested config support
    if isinstance(config.get("model"), dict):
        model_cfg = config["model"]
        data_cfg = config.get("data", {})
        in_channels = model_cfg.get("in_channels")
        if in_channels is None:
            in_channels = len(data_cfg.get("channels", [0]))
        return {
            "in_channels": in_channels,
            "num_classes": model_cfg.get("num_classes", 3),
            "base_channels": model_cfg.get("base_channels", 64),
            "depth": model_cfg.get("depth", 4),
            "use_attention": model_cfg.get("use_attention", False),
            "bilinear": model_cfg.get("bilinear", False),
        }

    # Flat config fallback
    in_channels = config.get("in_channels")
    if in_channels is None:
        in_channels = len(config.get("channels", [0]))
    return {
        "in_channels": in_channels,
        "num_classes": config.get("num_classes", 3),
        "base_channels": config.get("base_channels", 64),
        "depth": config.get("depth", 4),
        "use_attention": config.get("use_attention", False),
        "bilinear": config.get("bilinear", False),
    }


def _normalize_checkpoint_state_dict(state_dict: dict) -> dict:
    """Map legacy monolithic U-Net checkpoint keys to the current module layout."""
    legacy_prefixes = ("inc.", "encoders.", "decoders.", "outc.")
    if not any(key.startswith(legacy_prefixes) for key in state_dict):
        return state_dict

    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("inc."):
            new_key = "encoder.stem." + key.removeprefix("inc.")
        elif key.startswith("encoders."):
            new_key = "encoder.down_blocks." + key.removeprefix("encoders.")
        elif key.startswith("decoders."):
            new_key = "decoder.up_blocks." + key.removeprefix("decoders.")
        elif key.startswith("outc."):
            new_key = "decoder.outc." + key.removeprefix("outc.")
        else:
            new_key = key
        remapped[new_key] = value

    logger.info("Remapped legacy U-Net checkpoint keys to the current module layout")
    return remapped


def _channels_from_config(config: dict) -> list[int]:
    channels = _lookup(config, "data", "channels")
    if channels is not None:
        return channels
    return config.get("channels", [0])


class TBUNetPredictor:
    """
    Predictor class for TB U-Net inference.

    Provides a simple interface for running predictions on single images
    or batches, with proper preprocessing and postprocessing.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        channels: list[int] | None = None,
        pad_multiple: int = 32,
    ):
        """
        Initialize predictor.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            channels: Which channels to use (default: [0] for Phase)
            pad_multiple: Pad images to multiple of this value
        """
        self.model, self.config = load_model(checkpoint_path, device)
        self.device = next(self.model.parameters()).device
        self.channels = channels if channels is not None else _channels_from_config(self.config)
        self.pad_multiple = pad_multiple

    def _preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, dict]:
        """
        Preprocess image for inference.

        Args:
            image: Input image array (C, H, W) or (H, W)

        Returns:
            Tuple of (preprocessed tensor, padding info)
        """
        # Handle 2D input
        if image.ndim == 2:
            image = image[np.newaxis, ...]

        if image.ndim != 3:
            raise ValueError(f"Expected image with shape (C, H, W) or (H, W), got {image.shape}")

        # Select configured channels and zero-fill missing ones so checkpoints
        # trained with more channels can still run on narrower stacks.
        requested_channels = [int(c) for c in self.channels] if self.channels else [0]
        if not requested_channels:
            requested_channels = [0]

        source_count = int(image.shape[0])
        selected: list[np.ndarray] = []
        missing_channels: list[int] = []
        fallback = image[0]
        for channel_idx in requested_channels:
            if 0 <= channel_idx < source_count:
                selected.append(image[channel_idx])
            else:
                selected.append(np.zeros_like(fallback))
                missing_channels.append(channel_idx)

        if missing_channels:
            logger.warning(
                "Input stack has {} channel(s); zero-filling missing requested channels {}",
                source_count,
                missing_channels,
            )

        image = np.stack(selected, axis=0)

        # Normalize to [0, 1]
        image = image.astype(np.float32)
        for c in range(image.shape[0]):
            channel = image[c]
            cmin, cmax = channel.min(), channel.max()
            if cmax > cmin:
                image[c] = (channel - cmin) / (cmax - cmin)
            else:
                image[c] = 0.0

        # Pad to multiple
        _, h, w = image.shape
        new_h = ((h + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        new_w = ((w + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple

        pad_h = new_h - h
        pad_w = new_w - w

        if pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="reflect",
            )

        pad_info = {"original_h": h, "original_w": w, "pad_h": pad_h, "pad_w": pad_w}

        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image).unsqueeze(0)

        return tensor, pad_info

    def _postprocess(
        self,
        output: torch.Tensor,
        pad_info: dict,
        return_probs: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Postprocess model output.

        Args:
            output: Model logits (B, C, H, W)
            pad_info: Padding info from preprocessing
            return_probs: Whether to also return class probabilities

        Returns:
            Class predictions (H, W) or tuple of (predictions, probabilities)
        """
        # Remove batch dimension
        output = output[0]

        # Get predictions and probabilities
        probs = torch.softmax(output, dim=0).cpu().numpy()
        preds = np.argmax(probs, axis=0)

        # Remove padding
        h, w = pad_info["original_h"], pad_info["original_w"]
        preds = preds[:h, :w]
        probs = probs[:, :h, :w]

        if return_probs:
            return preds, probs
        return preds

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray | str | Path,
        return_probs: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Run prediction on a single image.

        Args:
            image: Input image array (C, H, W), (H, W), or path to TIFF
            return_probs: Whether to return class probabilities

        Returns:
            Predicted class labels (H, W) as uint8
            If return_probs=True, also returns (H, W, 3) probability array
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = tifffile.imread(str(image))

        # Preprocess
        tensor, pad_info = self._preprocess(image)
        tensor = tensor.to(self.device)

        # Forward pass
        logits = self.model(tensor)

        # Postprocess
        return self._postprocess(logits, pad_info, return_probs)

    def predict_cell_mask(
        self,
        image: np.ndarray | str | Path,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Run prediction and return binary cell mask.

        This is designed for integration with the existing pipeline.
        Returns a mask where cell pixels are labeled 1, everything else is 0.

        Args:
            image: Input image
            threshold: Minimum probability to consider a pixel as cell

        Returns:
            Binary mask (H, W) as uint16 (1 for cell, 0 otherwise)
        """
        preds, probs = self.predict(image, return_probs=True)

        # Get cell probability
        cell_prob = probs[1]  # Class 1 is cell

        # Create cell mask (cell class with sufficient probability)
        cell_mask = ((preds == 1) & (cell_prob >= threshold)).astype(np.uint16)

        # Label connected components
        if cell_mask.any():
            from scipy import ndimage
            labeled, n_labels = ndimage.label(cell_mask)
            # Keep only the largest connected component (central cell)
            if n_labels > 1:
                sizes = ndimage.sum(cell_mask, labeled, range(1, n_labels + 1))
                largest_label = np.argmax(sizes) + 1
                cell_mask = (labeled == largest_label).astype(np.uint16)
            else:
                cell_mask = (labeled > 0).astype(np.uint16)

        return cell_mask

    def classify_object(
        self,
        image: np.ndarray | str | Path,
        mask: np.ndarray | None = None,
    ) -> tuple[Literal["cell", "non_cell", "background"], float]:
        """
        Classify the central object in an image.

        Args:
            image: Input image
            mask: Optional existing mask to use for object location

        Returns:
            Tuple of (predicted class name, confidence score)
        """
        preds, probs = self.predict(image, return_probs=True)

        # Define region of interest
        if mask is not None and mask.any():
            roi_mask = mask > 0
        else:
            # Use center region
            h, w = preds.shape
            cy, cx = h // 2, w // 2
            margin = min(h, w) // 4
            roi_mask = np.zeros_like(preds, dtype=bool)
            roi_mask[cy - margin:cy + margin, cx - margin:cx + margin] = True

        # Get mean probabilities in ROI
        mean_probs = np.array([
            probs[c][roi_mask].mean() if roi_mask.any() else 0.0
            for c in range(3)
        ])

        # Classify
        pred_class = np.argmax(mean_probs)
        confidence = mean_probs[pred_class]

        class_names = {0: "background", 1: "cell", 2: "non_cell"}

        return class_names[pred_class], confidence


def predict(
    image: np.ndarray | str | Path,
    checkpoint_path: str | Path,
    device: str | None = None,
    return_probs: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for one-off predictions.

    Args:
        image: Input image array or path
        checkpoint_path: Path to model checkpoint
        device: Device to run on
        return_probs: Whether to return probabilities

    Returns:
        Predicted class labels (H, W)
    """
    predictor = TBUNetPredictor(checkpoint_path, device=device)
    return predictor.predict(image, return_probs=return_probs)


# Integration with SegmentationService
def create_unet_segment_function(
    checkpoint_path: str | Path,
    device: str | None = None,
):
    """
    Create a segment function compatible with SegmentationService.

    This allows the TB U-Net to be used as a drop-in replacement
    for MOMIA2 segmentation.

    Usage:
        from tb_unet.inference import create_unet_segment_function

        segment_fn = create_unet_segment_function("checkpoints/best.pt")
        labeled_mask = segment_fn(phase_image)

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run on

    Returns:
        Function that takes (H, W) image and returns labeled mask
    """
    predictor = TBUNetPredictor(checkpoint_path, device=device)

    def segment(image: np.ndarray) -> np.ndarray:
        """
        Segment an image and return labeled mask.

        Args:
            image: Phase image (H, W) or stack (C, H, W)

        Returns:
            Labeled mask (H, W) with cell pixels labeled 1
        """
        return predictor.predict_cell_mask(image)

    return segment
