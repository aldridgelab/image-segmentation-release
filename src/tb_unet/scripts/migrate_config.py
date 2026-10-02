#!/usr/bin/env python3
"""Migrate pre-refactor flat TB U-Net config YAML to nested format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tb_unet.config import TrainConfig

FLAT_TO_NESTED_KEY = {
    "data_dir": "data.data_dir",
    "image_dir": "data.image_dir",
    "channels": "data.channels",
    "pad_multiple": "data.pad_multiple",
    "in_channels": "model.in_channels",
    "num_classes": "model.num_classes",
    "base_channels": "model.base_channels",
    "depth": "model.depth",
    "use_attention": "model.use_attention",
    "bilinear": "model.bilinear",
    "batch_size": "training.batch_size",
    "num_workers": "training.num_workers",
    "epochs": "training.epochs",
    "lr": "training.lr",
    "weight_decay": "training.weight_decay",
    "scheduler": "training.scheduler",
    "scheduler_patience": "training.scheduler_patience",
    "scheduler_factor": "training.scheduler_factor",
    "early_stopping_patience": "training.early_stopping_patience",
    "mixed_precision": "training.mixed_precision",
    "gradient_clip": "training.gradient_clip",
    "loss_type": "loss.loss_type",
    "ce_weight": "loss.ce_weight",
    "dice_weight": "loss.dice_weight",
    "focal_gamma": "loss.focal_gamma",
    "class_weights": "loss.class_weights",
    "class_weight_method": "loss.class_weight_method",
    "output_dir": "output.output_dir",
    "experiment_name": "output.experiment_name",
    "save_every": "output.save_every",
    "log_every": "output.log_every",
    "wandb_enabled": "logging.wandb_enabled",
    "wandb_project": "logging.wandb_project",
    "wandb_entity": "logging.wandb_entity",
    "wandb_tags": "logging.wandb_tags",
}

NESTED_SECTIONS = {
    "data",
    "model",
    "training",
    "loss",
    "output",
    "logging",
    "augmentations",
}


def _is_nested_config(config: dict[str, Any]) -> bool:
    return any(section in config for section in NESTED_SECTIONS)


def _set_nested(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def migrate_flat_config(flat: dict[str, Any]) -> dict[str, Any]:
    nested = TrainConfig().to_dict()

    for old_key, new_key in FLAT_TO_NESTED_KEY.items():
        if old_key in flat:
            _set_nested(nested, new_key, flat[old_key])

    if "run_id" in flat:
        nested["run_id"] = flat["run_id"]

    # Preserve explicitly nested sections if user partially migrated already.
    for section in NESTED_SECTIONS:
        if section in flat and isinstance(flat[section], dict):
            for key, value in flat[section].items():
                _set_nested(nested, f"{section}.{key}", value)

    return TrainConfig.from_dict(nested).to_dict()


def migrate_yaml_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and "experiments" in payload and isinstance(payload["experiments"], list):
        migrated = []
        for exp in payload["experiments"]:
            if not isinstance(exp, dict):
                raise TypeError("Each experiment must be a mapping")
            migrated.append(exp if _is_nested_config(exp) else migrate_flat_config(exp))
        return {"experiments": migrated}

    if isinstance(payload, dict):
        return payload if _is_nested_config(payload) else migrate_flat_config(payload)

    raise TypeError("Input YAML must be a config mapping or {'experiments': [...]} structure")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate flat TB U-Net config to nested format")
    parser.add_argument("--input", required=True, help="Path to input YAML")
    parser.add_argument("--output", required=True, help="Path to output YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate migration without writing output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r") as f:
        payload = yaml.safe_load(f) or {}

    migrated = migrate_yaml_payload(payload)

    if args.dry_run:
        print(f"[DRY RUN] Migration validated for: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(migrated, f, sort_keys=False)

    print(f"Migrated config written to: {output_path}")


if __name__ == "__main__":
    main()
