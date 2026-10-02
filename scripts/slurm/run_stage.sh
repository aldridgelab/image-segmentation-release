#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$1"
CONFIG="$2"
STAGE="$3"
BATCH_NAME="${4:-}"
SHARD="${5:-}"

cd "$REPO_ROOT"

cmd=(uv run python pipeline.py --config "$CONFIG" --stage "$STAGE")
if [[ -n "$BATCH_NAME" ]]; then
  cmd+=(--batch-name "$BATCH_NAME")
fi
if [[ -n "$SHARD" ]]; then
  cmd+=(--shard "$SHARD")
fi

"${cmd[@]}"
