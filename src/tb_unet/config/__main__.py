"""CLI entry points for tb_unet.config."""

from __future__ import annotations

import argparse

from tb_unet.config.schema import generate_schema_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="TB U-Net config utilities")
    subparsers = parser.add_subparsers(dest="command")

    schema_parser = subparsers.add_parser("schema", help="Generate reference schema YAML")
    schema_parser.add_argument(
        "--output",
        default="tb_unet/configs/schema_reference.yaml",
        help="Path for the generated schema YAML",
    )

    args = parser.parse_args()

    if args.command == "schema":
        path = generate_schema_yaml(args.output)
        print(f"Wrote schema reference: {path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
