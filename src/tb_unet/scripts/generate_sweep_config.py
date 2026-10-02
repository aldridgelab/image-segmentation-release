#!/usr/bin/env python3
"""
Generate hyperparameter sweep configurations for TB U-Net.

Creates YAML config files for grid search or random search sweeps.
Each generated config is compatible with `python -m tb_unet.train`.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import random
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tb_unet.config import TrainConfig

# Backward-compat aliases for older flat templates.
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


def load_sweep_template(template_path: Path) -> dict[str, Any]:
    """Load sweep parameter template."""
    with template_path.open("r") as f:
        return yaml.safe_load(f) or {}


def _canonical_key(key: str) -> str:
    if "." in key:
        return key
    return FLAT_TO_NESTED_KEY.get(key, key)


def _set_nested(config: dict, key: str, value: Any) -> None:
    """Set a nested config value using dot notation."""
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _get_nested(config: dict[str, Any], key: str, default: Any = None) -> Any:
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _fmt_value(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:g}"
    if isinstance(val, list):
        return "".join(str(v) for v in val)
    if isinstance(val, bool):
        return "1" if val else "0"
    return str(val)


def _format_run_id(config: dict[str, Any], group_name: str) -> str:
    """Build a descriptive run_id from key hyperparameters."""
    channels = _get_nested(config, "data.channels", [0])
    use_attention = _get_nested(config, "model.use_attention", False)
    base_channels = _get_nested(config, "model.base_channels", 64)
    depth = _get_nested(config, "model.depth", 4)
    lr = _get_nested(config, "training.lr", 0.001)
    focal_gamma = _get_nested(config, "loss.focal_gamma", 2.0)
    loss_type = _get_nested(config, "loss.loss_type", "combined")

    parts = [
        group_name,
        f"ch{_fmt_value(channels)}",
        f"att{_fmt_value(use_attention)}",
        f"bc{_fmt_value(base_channels)}",
        f"d{_fmt_value(depth)}",
        f"lr{_fmt_value(lr)}",
        f"fg{_fmt_value(focal_gamma)}",
        str(loss_type).replace("_", ""),
    ]

    return "_".join(parts).replace("/", "-").replace(" ", "")


def _build_base_config(template: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    """Build base nested config from defaults, template overrides, and fixed values."""
    config: dict[str, Any] = TrainConfig().to_dict()

    # Nested section overrides from template.
    for section in [
        "data",
        "model",
        "training",
        "loss",
        "output",
        "logging",
        "augmentations",
    ]:
        section_values = template.get(section)
        if isinstance(section_values, dict):
            for key, value in section_values.items():
                _set_nested(config, f"{section}.{key}", copy.deepcopy(value))

    # Flat key overrides from template (backward compatible).
    for key, value in template.items():
        if key in {"run_name", "group_name", "parameters", "fixed"}:
            continue
        nested_key = _canonical_key(key)
        if nested_key == key and "." not in nested_key:
            continue
        _set_nested(config, nested_key, copy.deepcopy(value))

    # Fixed values are sweep-global constants.
    for name, value in fixed.items():
        _set_nested(config, _canonical_key(name), copy.deepcopy(value))

    # Ensure group tag exists unless explicitly configured.
    group_name = template.get("group_name", "sweep")
    tags = _get_nested(config, "logging.wandb_tags", [])
    if not tags:
        _set_nested(config, "logging.wandb_tags", [group_name])

    return config


def _apply_parameters(
    config: dict[str, Any],
    param_names: list[str],
    combo: tuple[Any, ...],
) -> None:
    for name, value in zip(param_names, combo):
        _set_nested(config, _canonical_key(name), value)


def _finalize_config(config: dict[str, Any], group_name: str) -> dict[str, Any]:
    run_id = _format_run_id(config, group_name)
    config["run_id"] = run_id
    _set_nested(config, "output.experiment_name", run_id)

    # Keep in_channels inferred unless explicitly set.
    if _get_nested(config, "model.in_channels") is None:
        channels = _get_nested(config, "data.channels", [0])
        _set_nested(config, "model.in_channels", len(channels))

    # Validate by round-tripping through TrainConfig.
    validated = TrainConfig.from_dict(config)
    return validated.to_dict()


def generate_grid_sweep(
    parameters: dict[str, list[Any]],
    fixed: dict[str, Any],
    template: dict[str, Any],
    output_dir: Path,
    write_individual: bool = False,
) -> list[dict[str, Any]]:
    """Generate grid search configurations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    group_name = template.get("group_name", "sweep")

    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    combinations = list(itertools.product(*param_values))

    configs: list[dict[str, Any]] = []
    for i, combo in enumerate(combinations):
        config = _build_base_config(template, fixed)
        _apply_parameters(config, param_names, combo)
        config = _finalize_config(config, group_name)

        if write_individual:
            config_path = output_dir / f"run_{i:04d}.yaml"
            with config_path.open("w") as f:
                yaml.safe_dump(config, f, sort_keys=False)

        configs.append(config)

    logger.info(f"Generated {len(configs)} grid sweep configs")
    return configs


def generate_random_sweep(
    parameters: dict[str, list[Any]],
    fixed: dict[str, Any],
    template: dict[str, Any],
    output_dir: Path,
    n_runs: int = 20,
    seed: int = 42,
    write_individual: bool = False,
) -> list[dict[str, Any]]:
    """Generate random search configurations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    group_name = template.get("group_name", "sweep")
    param_names = list(parameters.keys())

    configs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    attempts = 0
    max_attempts = n_runs * 10

    while len(configs) < n_runs and attempts < max_attempts:
        attempts += 1
        config = _build_base_config(template, fixed)

        sampled_values = tuple(random.choice(parameters[name]) for name in param_names)
        _apply_parameters(config, param_names, sampled_values)
        config = _finalize_config(config, group_name)

        run_id = config["run_id"]
        if run_id in seen_ids:
            continue

        seen_ids.add(run_id)

        if write_individual:
            config_path = output_dir / f"run_{len(configs):04d}.yaml"
            with config_path.open("w") as f:
                yaml.safe_dump(config, f, sort_keys=False)

        configs.append(config)

    logger.info(f"Generated {len(configs)} random sweep configs")
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TB U-Net sweep configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--template",
        type=str,
        default="tb_unet/configs/sweeps/sweep_params.yaml",
        help="Sweep template YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tb_unet/configs_generated",
        help="Output directory for generated configs",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "random"],
        default="grid",
        help="Sweep method: grid (all combinations) or random (sample)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=50,
        help="Number of runs for random sweep (ignored for grid)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save per-run YAML files in addition to the bundled file",
    )

    args = parser.parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        logger.error(f"Template not found: {template_path}")
        return

    template = load_sweep_template(template_path)
    parameters = template.get("parameters", {})
    fixed = template.get("fixed", {})

    if not parameters:
        logger.error("No parameters found in template")
        return

    output_dir = Path(args.output_dir)
    bundle_name = template.get("run_name", "sweep")

    if args.method == "grid":
        configs = generate_grid_sweep(
            parameters=parameters,
            fixed=fixed,
            template=template,
            output_dir=output_dir,
            write_individual=args.save_individual,
        )
    else:
        configs = generate_random_sweep(
            parameters=parameters,
            fixed=fixed,
            template=template,
            output_dir=output_dir,
            n_runs=args.n_runs,
            seed=args.seed,
            write_individual=args.save_individual,
        )

    if configs:
        bundle_path = output_dir / f"{bundle_name}_bundle.yaml"
        output_dir.mkdir(parents=True, exist_ok=True)
        with bundle_path.open("w") as f:
            yaml.safe_dump({"experiments": configs}, f, sort_keys=False)
        logger.info(f"Bundled {len(configs)} configs into {bundle_path}")

        logger.info("\nSweep Summary:")
        logger.info(f"  Method: {args.method}")
        logger.info(f"  Total runs: {len(configs)}")
        logger.info(f"  Output: {bundle_path}")

        if args.method == "grid":
            logger.info("\n  Parameter grid:")
            for name, values in parameters.items():
                logger.info(f"    {name}: {values}")


if __name__ == "__main__":
    main()
