#!/usr/bin/env bash
set -e

# Edit these values for each sample.
VIDEO="/path/to/full_or_adjacent_shots.mp4"
PREV_END="00:00:05.000"
START="00:00:05.000"
END="00:00:10.000"
PROMPT="The character continues walking into the room."
OUTPUT="data/storymem_cut_false_mi2v"

# Usually keep these defaults unless the training config changes.
NUM_FRAMES=49
FPS=16
WIDTH=832
HEIGHT=480
QUALITY=8

python examples/wanvideo/model_training/scripts/make_storymem_cut_false_dataset.py \
  --video "${VIDEO}" \
  --prev-end "${PREV_END}" \
  --start "${START}" \
  --end "${END}" \
  --prompt "${PROMPT}" \
  --output "${OUTPUT}" \
  --num-frames "${NUM_FRAMES}" \
  --fps "${FPS}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --quality "${QUALITY}" \
  --overwrite
