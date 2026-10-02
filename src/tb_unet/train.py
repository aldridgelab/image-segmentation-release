"""Training script for TB U-Net with nested config support."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from tb_unet.config import TrainConfig
from tb_unet.dataset import create_dataloaders
from tb_unet.losses import CombinedFocalDiceLoss, CombinedLoss, compute_class_weights
from tb_unet.metrics import MetricsCalculator, print_metrics
from tb_unet.model import build_unet


class Trainer:
    """
    Trainer for TB U-Net.

    Handles training loop, validation, logging, and checkpointing.
    """

    def __init__(self, config: TrainConfig, *, wandb: bool | None = None):
        self.config = config
        self.config.validate()

        if wandb is not None:
            self.config.logging.wandb_enabled = wandb

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(self.config.output.output_dir) / self.config.output.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save resolved config
        self.config.to_yaml(self.output_dir / "config.yaml")

        # Initialize W&B
        self.wandb_run = None
        if self.config.logging.wandb_enabled:
            self._init_wandb()

        # Create dataloaders
        logger.info("Creating dataloaders...")
        self.dataloaders = create_dataloaders(
            data_dir=self.config.data.data_dir,
            image_dir=self.config.data.image_dir,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            channels=self.config.data.channels,
            pad_multiple=self.config.data.pad_multiple,
            augmentations_config=self.config.augmentations,
        )

        # Compute class weights if not provided
        if self.config.loss.class_weights is None:
            logger.info("Computing class weights from training data...")
            self.config.loss.class_weights = self._compute_class_weights()
            logger.info(f"Class weights: {self.config.loss.class_weights}")

        # Create model
        logger.info("Creating model...")
        self.model = build_unet(self.config.model)
        self.model.to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

        # Create loss/optimizer/scheduler
        self.criterion = self._create_loss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if self.config.training.mixed_precision else None

        # Training state
        self.epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0

        # Local history
        self.history = {
            "train/loss": [],
            "val/loss": [],
            "val/dice_mean": [],
            "val/dice_cell": [],
            "val/dice_non_cell": [],
            "val/iou_mean": [],
            "val/cell_precision": [],
            "val/cell_recall": [],
            "val/cell_f1": [],
            "val/non_cell_tnr": [],
            "lr": [],
        }

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
                name=self.config.output.experiment_name,
                config=self.config.to_dict(),
                tags=self.config.logging.wandb_tags,
                dir=str(self.output_dir),
            )
            logger.info(f"W&B initialized: {wandb.run.url}")
        except ImportError:
            logger.warning("wandb not installed, disabling logging")
            self.config.logging.wandb_enabled = False
        except Exception as exc:
            logger.warning(f"Failed to initialize W&B: {exc}")
            self.config.logging.wandb_enabled = False

    def _compute_class_weights(self) -> list[float]:
        """Compute class weights from training data distribution."""
        class_counts = [0 for _ in range(self.config.model.num_classes)]

        for batch in tqdm(self.dataloaders["train"], desc="Computing class weights"):
            labels = batch["label"].numpy()
            for class_id in range(self.config.model.num_classes):
                class_counts[class_id] += (labels == class_id).sum()

        logger.info(
            "Class counts: "
            + ", ".join(f"class_{i}={count}" for i, count in enumerate(class_counts))
        )

        return compute_class_weights(class_counts, method=self.config.loss.class_weight_method)

    def _create_loss(self) -> nn.Module:
        """Create loss function based on config."""
        if self.config.loss.loss_type == "combined":
            return CombinedLoss(
                ce_weight=self.config.loss.ce_weight,
                dice_weight=self.config.loss.dice_weight,
                class_weights=self.config.loss.class_weights,
            )

        if self.config.loss.loss_type == "focal_dice":
            return CombinedFocalDiceLoss(
                focal_weight=self.config.loss.ce_weight,
                dice_weight=self.config.loss.dice_weight,
                alpha=self.config.loss.class_weights,
                gamma=self.config.loss.focal_gamma,
                class_weights=self.config.loss.class_weights,
            )

        raise ValueError(f"Unknown loss type: {self.config.loss.loss_type}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config.training.scheduler

        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=1e-6,
            )

        if scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
            )

        if scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )

        return None

    def train_epoch(self) -> dict:
        """Run one training epoch."""
        self.model.train()
        metrics_calc = MetricsCalculator()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.dataloaders["train"],
            desc=f"Epoch {self.epoch + 1}/{self.config.training.epochs} [Train]",
        )

        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            class_labels = batch["class_label"]

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    logits = self.model(images)
                    loss_dict = self.criterion(logits, labels)
                    loss = loss_dict["loss"]

                self.scaler.scale(loss).backward()

                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss_dict = self.criterion(logits, labels)
                loss = loss_dict["loss"]
                loss.backward()

                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip
                    )

                self.optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                metrics_calc.update(preds, labels, class_labels)

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        metrics = metrics_calc.summary()
        metrics["loss"] = avg_loss
        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation."""
        self.model.eval()
        metrics_calc = MetricsCalculator()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.dataloaders["val"],
            desc=f"Epoch {self.epoch + 1}/{self.config.training.epochs} [Val]",
        )

        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            class_labels = batch["class_label"]

            logits = self.model(images)
            loss_dict = self.criterion(logits, labels)
            loss = loss_dict["loss"]

            preds = torch.argmax(logits, dim=1)
            metrics_calc.update(preds, labels, class_labels)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        metrics = metrics_calc.summary()
        metrics["loss"] = avg_loss
        return metrics

    def train(self):
        """Run full training loop."""
        logger.info(f"Starting training for {self.config.training.epochs} epochs...")

        for epoch in range(self.config.training.epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler is not None:
                if self.config.training.scheduler == "plateau":
                    self.scheduler.step(val_metrics["cell_f1"])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_dice={val_metrics['dice_mean']:.4f}, "
                f"val_f1={val_metrics['cell_f1']:.4f}, "
                f"lr={current_lr:.2e}"
            )

            self.history["train/loss"].append(train_metrics["loss"])
            self.history["val/loss"].append(val_metrics["loss"])
            self.history["val/dice_mean"].append(val_metrics["dice_mean"])
            self.history["val/dice_cell"].append(val_metrics.get("dice_cell", 0))
            self.history["val/dice_non_cell"].append(val_metrics.get("dice_non_cell", 0))
            self.history["val/iou_mean"].append(val_metrics["iou_mean"])
            self.history["val/cell_precision"].append(val_metrics["cell_precision"])
            self.history["val/cell_recall"].append(val_metrics["cell_recall"])
            self.history["val/cell_f1"].append(val_metrics["cell_f1"])
            self.history["val/non_cell_tnr"].append(val_metrics["non_cell_tnr"])
            self.history["lr"].append(current_lr)

            self._save_history()

            if self.config.logging.wandb_enabled and self.wandb_run is not None:
                import wandb

                log_dict = {"epoch": epoch + 1, "lr": current_lr}
                for key, value in train_metrics.items():
                    log_dict[f"train/{key}"] = value
                for key, value in val_metrics.items():
                    log_dict[f"val/{key}"] = value
                wandb.log(log_dict)

            if (epoch + 1) % self.config.output.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            metric = val_metrics["cell_f1"]
            if metric > self.best_metric:
                self.best_metric = metric
                self.patience_counter = 0
                self.save_checkpoint("best.pt")
                logger.info(f"New best model! F1={metric:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(
                        f"Early stopping after {epoch + 1} epochs "
                        f"(no improvement for {self.patience_counter} epochs)"
                    )
                    break

        logger.info("Evaluating on test set...")
        self.load_checkpoint("best.pt")
        test_metrics = self._evaluate_test()

        if self.config.logging.wandb_enabled and self.wandb_run is not None:
            import wandb

            for key, value in test_metrics.items():
                wandb.summary[f"test/{key}"] = value
            wandb.finish()

        logger.info("Training complete!")
        return test_metrics

    @torch.no_grad()
    def _evaluate_test(self) -> dict:
        """Evaluate on test set."""
        self.model.eval()
        metrics_calc = MetricsCalculator()

        for batch in tqdm(self.dataloaders["test"], desc="Test evaluation"):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            class_labels = batch["class_label"]

            logits = self.model(images)
            preds = torch.argmax(logits, dim=1)
            metrics_calc.update(preds, labels, class_labels)

        metrics = metrics_calc.compute()
        print_metrics(metrics, "Test Set Results")
        return metrics_calc.summary()

    def _save_history(self) -> None:
        """Save training history to JSON file."""
        history_path = self.output_dir / "training_history.json"
        with history_path.open("w") as f:
            json.dump(self.history, f, indent=2)

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "best_metric": self.best_metric,
                "config": self.config.to_dict(),
            },
            path,
        )
        logger.debug(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self.output_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        logger.info(f"Loaded checkpoint: {path}")


def run_single_experiment(config: TrainConfig, *, wandb: bool | None = None) -> dict:
    """Run a single training experiment and return test metrics."""
    trainer = Trainer(config, wandb=wandb)
    return trainer.train()


def load_configs_from_yaml(path: str | Path) -> list[TrainConfig]:
    """
    Load one or more configs from YAML.

    Supports:
    1. Single nested config
    2. Batch config with top-level `experiments`
    """
    with Path(path).open("r") as f:
        data = yaml.safe_load(f) or {}

    if isinstance(data, dict) and "experiments" in data:
        experiments = data["experiments"]
        logger.info(f"Loaded batch config with {len(experiments)} experiments")
        return [TrainConfig.from_dict(exp) for exp in experiments]

    return [TrainConfig.from_dict(data)]


def main() -> None:
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train TB U-Net model")
    parser.add_argument(
        "--config",
        type=str,
        default="tb_unet/configs/default.yaml",
        help="Path to config YAML file (single or batch format)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    args = parser.parse_args()

    configs = load_configs_from_yaml(args.config)

    if args.no_wandb:
        for config in configs:
            config.logging.wandb_enabled = False

    for i, config in enumerate(configs, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"Starting experiment {i}/{len(configs)}: {config.output.experiment_name}"
        )
        logger.info(f"{'=' * 60}\n")

        try:
            run_single_experiment(config)
            logger.info(f"Experiment {i}/{len(configs)} completed successfully")
        except Exception as exc:
            logger.error(f"Experiment {i}/{len(configs)} failed: {exc}")
            continue

    logger.info(f"\nAll {len(configs)} experiments completed")


if __name__ == "__main__":
    main()
