#!/usr/bin/env bash
set -e

python examples/wanvideo/model_training/scripts/make_storymem_cut_false_dataset.py \
  --video data/6am.mp4 \
  --memory-start "00:00:00.000" \
  --prev-end "00:00:00.500" \
  --start "00:00:00.500" \
  --end "00:00:04.000" \
  --prompt "A man wakes up in bed, shifts restlessly, then grabs a pillow, rolls onto his side, and pulls it over his head." \
  --output data/6am \
  --num-frames 49 \
  --num-memory-images 3 \
  --fps 16 \
  --width 832 \
  --height 480 \
  --quality 8 \
  --overwrite
