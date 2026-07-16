#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare one cut=False StoryMem MI2V training sample.

Required environment variables:
  VIDEO=/path/to/full_or_adjacent_shots.mp4
  PREV_END=00:00:05.000
  START=00:00:05.000
  END=00:00:10.000
  PROMPT="The character continues walking into the room."

Optional environment variables:
  OUTPUT=data/storymem_cut_false_mi2v
  NUM_FRAMES=49
  FPS=16
  WIDTH=832
  HEIGHT=480
  QUALITY=8
  SAMPLE_NAME=sample_000000
  METADATA_NAME=metadata.csv
  OVERWRITE=1

Optional repeated inputs:
  MEMORY_TIMES="00:00:01.000 00:00:03.000"
  MEMORY_FRAMES="/path/to/a.png /path/to/b.png"

Example:
  VIDEO=/path/to/video.mp4 \
  PREV_END=00:00:05.000 \
  START=00:00:05.000 \
  END=00:00:10.000 \
  PROMPT="The character continues walking into the room." \
  OUTPUT=data/storymem_cut_false_mi2v \
  OVERWRITE=1 \
  bash examples/wanvideo/model_training/scripts/make_storymem_cut_false_dataset.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    echo >&2
    usage >&2
    exit 2
  fi
}

require_env VIDEO
require_env PREV_END
require_env START
require_env END
require_env PROMPT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/make_storymem_cut_false_dataset.py"
  --video "${VIDEO}"
  --prev-end "${PREV_END}"
  --start "${START}"
  --end "${END}"
  --prompt "${PROMPT}"
  --output "${OUTPUT:-data/storymem_cut_false_mi2v}"
  --num-frames "${NUM_FRAMES:-49}"
  --quality "${QUALITY:-8}"
  --metadata-name "${METADATA_NAME:-metadata.csv}"
)

if [[ -n "${FPS:-}" ]]; then
  cmd+=(--fps "${FPS}")
fi
if [[ -n "${WIDTH:-}" ]]; then
  cmd+=(--width "${WIDTH}")
fi
if [[ -n "${HEIGHT:-}" ]]; then
  cmd+=(--height "${HEIGHT}")
fi
if [[ -n "${SAMPLE_NAME:-}" ]]; then
  cmd+=(--sample-name "${SAMPLE_NAME}")
fi
if [[ "${OVERWRITE:-0}" == "1" || "${OVERWRITE:-}" == "true" || "${OVERWRITE:-}" == "TRUE" ]]; then
  cmd+=(--overwrite)
fi

if [[ -n "${MEMORY_TIMES:-}" ]]; then
  for memory_time in ${MEMORY_TIMES}; do
    cmd+=(--memory-time "${memory_time}")
  done
fi
if [[ -n "${MEMORY_FRAMES:-}" ]]; then
  for memory_frame in ${MEMORY_FRAMES}; do
    cmd+=(--memory-frame "${memory_frame}")
  done
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
