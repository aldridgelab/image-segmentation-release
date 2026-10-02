"""Utilities for generating reference schema YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from tb_unet.config.train import TrainConfig


def generate_schema_yaml(output_path: str | Path = "tb_unet/configs/schema_reference.yaml") -> Path:
    """Write a reference config YAML with section comments and defaults."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    defaults = TrainConfig().to_dict()
    lines = [
        "# TB U-Net nested config reference",
        "# Edit a copy of this file for training runs.",
        "",
    ]

    section_descriptions = {
        "data": "Input paths, channel selection, and padding behavior.",
        "model": "U-Net architecture settings.",
        "training": "Optimizer/scheduler/training loop settings.",
        "loss": "Loss composition and class weighting.",
        "output": "Checkpoint/log output paths and cadence.",
        "logging": "W&B settings and tags.",
        "augmentations": "Transform strategy and per-transform probabilities.",
    }

    for section in ["data", "model", "training", "loss", "output", "logging", "augmentations"]:
        lines.append(f"# == {section.upper()} ==")
        lines.append(f"# {section_descriptions[section]}")
        rendered = yaml.safe_dump({section: defaults[section]}, sort_keys=False).strip()
        lines.append(rendered)
        lines.append("")

    lines.append("run_id: null")
    output.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate TB U-Net config schema YAML")
    parser.add_argument("--output", default="tb_unet/configs/schema_reference.yaml")
    args = parser.parse_args()

    path = generate_schema_yaml(args.output)
    print(f"Wrote schema reference: {path}")


if __name__ == "__main__":
    main()
