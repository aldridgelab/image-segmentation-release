#!/bin/bash
# Training script for TB U-Net
# Usage: ./scripts/train.sh [config_name]
#
# Examples:
#   ./scripts/train.sh                  # Uses default.yaml
#   ./scripts/train.sh focal_loss       # Uses focal_loss.yaml
#   ./scripts/train.sh custom_name      # Uses custom_name.yaml

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
TBUNET_DIR="$(dirname "$SCRIPT_DIR")"

# Default config
CONFIG_NAME="${1:-default}"
CONFIG_PATH="$TBUNET_DIR/configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    echo "Available configs:"
    ls -1 "$TBUNET_DIR/configs/"*.yaml | xargs -n1 basename
    exit 1
fi

echo "=============================================="
echo "TB U-Net Training"
echo "=============================================="
echo "Config: $CONFIG_PATH"
echo "Project: $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

# Run training
PYTHONPATH="$PROJECT_DIR" uv run python -m tb_unet.train --config "$CONFIG_PATH" "${@:2}"
