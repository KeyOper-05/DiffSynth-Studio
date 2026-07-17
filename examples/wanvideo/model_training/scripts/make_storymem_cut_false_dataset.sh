#!/usr/bin/env bash
set -e

# Edit these values for each sample.
VIDEO="data/6am.mp4"
PREV_END="00:00:00.000"
START="00:00:00.000"
END="00:00:04.000"
PROMPT="A man wakes up in bed, shifts restlessly, then grabs a pillow, rolls onto his side, and pulls it over his head."
OUTPUT="data/6am"

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
