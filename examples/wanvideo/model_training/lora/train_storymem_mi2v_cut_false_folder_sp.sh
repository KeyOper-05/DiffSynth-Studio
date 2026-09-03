#!/usr/bin/env bash
set -euo pipefail

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
  echo "Usage: bash $0 <dataset_dir> <storymem_lora_output_dir>"
  echo "Example: bash $0 data/6am ../StoryMem/examples/motion_loras/6am_sp"
  exit 1
fi

DATASET_DIR="$1"
OUTPUT_DIR="$2"
METADATA_PATH="$DATASET_DIR/metadata.csv"
HIGH_OUTPUT_DIR="$OUTPUT_DIR/diffsynth_high_noise_lora"
LOW_OUTPUT_DIR="$OUTPUT_DIR/diffsynth_low_noise_lora"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-49}"
DATASET_REPEAT="${DATASET_REPEAT:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
TRAINING_SCHEDULER_SHIFT="${TRAINING_SCHEDULER_SHIFT:-4.0}"
TIMESTEP_BOUNDARY="${TIMESTEP_BOUNDARY:-0.308}"
SP_SIZE="${SP_SIZE:-8}"
SP_SEED="${SP_SEED:-0}"
LAST_EPOCH=$((NUM_EPOCHS - 1))

if [ ! -f "$METADATA_PATH" ]; then
  echo "Missing metadata file: $METADATA_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1
export WANDB_PROJECT="${WANDB_PROJECT:-StoryMem-LoRA-SP}"
RUN_PREFIX="${WANDB_RUN_PREFIX:-$(basename "$OUTPUT_DIR")}-sp${SP_SIZE}"

build_model_paths() {
  local noise_stage="$1"
  local shards=""
  local index
  local shard
  for index in 1 2 3 4 5 6; do
    printf -v shard "%05d" "$index"
    if [ -n "$shards" ]; then
      shards+=","
    fi
    shards+="\"./models/storymem_prefused/Wan2.2-MI2V-A14B/${noise_stage}_noise_model/diffusion_pytorch_model-${shard}-of-00006.safetensors\""
  done
  printf '[[%s],"../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth","../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth"]' "$shards"
}

run_stage() {
  local noise_stage="$1"
  local min_boundary="$2"
  local max_boundary="$3"
  local stage_output="$4"
  local model_paths
  model_paths="$(build_model_paths "$noise_stage")"

  WANDB_NAME="${RUN_PREFIX}-mi2v-${noise_stage}-noise" accelerate launch \
    --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
    examples/wanvideo/model_training/train_sequence_parallel.py \
    --sequence_parallel_size "$SP_SIZE" \
    --sequence_parallel_seed "$SP_SEED" \
    --dataset_base_path "$DATASET_DIR" \
    --dataset_metadata_path "$METADATA_PATH" \
    --data_file_keys "video,memory_images,input_image" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_frames "$NUM_FRAMES" \
    --dataset_repeat "$DATASET_REPEAT" \
    --model_paths "$model_paths" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "$stage_output" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank "$LORA_RANK" \
    --extra_inputs "memory_images,input_image" \
    --training_scheduler_shift "$TRAINING_SCHEDULER_SHIFT" \
    --max_timestep_boundary "$max_boundary" \
    --min_timestep_boundary "$min_boundary" \
    --enable_wandb_log \
    --wandb_project "$WANDB_PROJECT" \
    --initialize_model_on_cpu
}

run_stage high 0 "$TIMESTEP_BOUNDARY" "$HIGH_OUTPUT_DIR"
run_stage low "$TIMESTEP_BOUNDARY" 1 "$LOW_OUTPUT_DIR"

HIGH_EXPORT="$HIGH_OUTPUT_DIR/epoch-${LAST_EPOCH}.safetensors"
LOW_EXPORT="$LOW_OUTPUT_DIR/epoch-${LAST_EPOCH}.safetensors"

if [ ! -f "$HIGH_EXPORT" ] || [ ! -f "$LOW_EXPORT" ]; then
  echo "Missing expected LoRA checkpoint(s):"
  echo "  $HIGH_EXPORT"
  echo "  $LOW_EXPORT"
  exit 1
fi

cp "$HIGH_EXPORT" "$OUTPUT_DIR/backbone_high_noise.safetensors"
cp "$LOW_EXPORT" "$OUTPUT_DIR/backbone_low_noise.safetensors"

echo "Sequence-parallel StoryMem LoRA folder is ready: $OUTPUT_DIR"
