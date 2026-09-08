#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a

cd "$(dirname "$0")"

ACTION_NAME="${1:-}"

if [ -z "$ACTION_NAME" ]; then
  echo "Usage: bash train_storymem_cut_false.sh <action_name>"
  echo "Example: bash train_storymem_cut_false.sh 6am"
  exit 1
fi

if [ ! -f "data/${ACTION_NAME}/metadata.csv" ]; then
  echo "Missing training metadata: data/${ACTION_NAME}/metadata.csv"
  exit 1
fi

echo "Starting StoryMem cut=False training: ${ACTION_NAME}"
echo "W&B project: ${WANDB_PROJECT:-StoryMem-LoRA}"

bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-MI2V-cut-false-A14B.sh "$ACTION_NAME"
