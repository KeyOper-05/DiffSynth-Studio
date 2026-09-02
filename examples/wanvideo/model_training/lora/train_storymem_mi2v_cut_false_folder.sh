#!/usr/bin/env bash
set -euo pipefail

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
  echo "Usage: bash $0 <dataset_dir> <storymem_lora_output_dir> [storymem|mi2v_cut_false]"
  echo "Example: bash $0 data/6am ../StoryMem/examples/motion_loras/6am storymem"
  exit 1
fi

DATASET_DIR="$1"
OUTPUT_DIR="$2"
TRAIN_MODE="${3:-mi2v_cut_false}"
METADATA_PATH="$DATASET_DIR/metadata.csv"
HIGH_OUTPUT_DIR="$OUTPUT_DIR/diffsynth_high_noise_lora"
LOW_OUTPUT_DIR="$OUTPUT_DIR/diffsynth_low_noise_lora"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LAST_EPOCH=$((NUM_EPOCHS - 1))
HIGH_EXPORT="$HIGH_OUTPUT_DIR/epoch-${LAST_EPOCH}.safetensors"
LOW_EXPORT="$LOW_OUTPUT_DIR/epoch-${LAST_EPOCH}.safetensors"

case "$TRAIN_MODE" in
  storymem)
    DATA_FILE_KEYS="video,memory_images"
    EXTRA_INPUTS="memory_images"
    LORA_RANK="${LORA_RANK:-32}"
    HIGH_BOUNDARY="${TIMESTEP_BOUNDARY:-0.1}"
    LOW_BOUNDARY="$HIGH_BOUNDARY"
    SCHEDULER_ARGS=()
    RUN_KIND="t2v"
    ;;
  mi2v_cut_false)
    DATA_FILE_KEYS="video,memory_images,input_image"
    EXTRA_INPUTS="memory_images,input_image"
    LORA_RANK="${LORA_RANK:-16}"
    TRAINING_SCHEDULER_SHIFT="${TRAINING_SCHEDULER_SHIFT:-4.0}"
    HIGH_BOUNDARY="${TIMESTEP_BOUNDARY:-0.308}"
    LOW_BOUNDARY="$HIGH_BOUNDARY"
    SCHEDULER_ARGS=(--training_scheduler_shift "$TRAINING_SCHEDULER_SHIFT")
    RUN_KIND="mi2v"
    ;;
  *)
    echo "Unsupported train mode: $TRAIN_MODE (expected storymem or mi2v_cut_false)"
    exit 1
    ;;
esac

if [ ! -f "$METADATA_PATH" ]; then
  echo "Missing metadata file: $METADATA_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1
export WANDB_PROJECT="${WANDB_PROJECT:-StoryMem-LoRA}"
RUN_PREFIX="${WANDB_RUN_PREFIX:-$(basename "$OUTPUT_DIR")}"

WANDB_NAME="${RUN_PREFIX}-${RUN_KIND}-high-noise" accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_DIR" \
  --dataset_metadata_path "$METADATA_PATH" \
  --data_file_keys "$DATA_FILE_KEYS" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "[[\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"],\"../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth\",\"../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth\"]" \
  --learning_rate 1e-4 \
  --num_epochs "$NUM_EPOCHS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$HIGH_OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "$LORA_RANK" \
  --extra_inputs "$EXTRA_INPUTS" \
  "${SCHEDULER_ARGS[@]}" \
  --max_timestep_boundary "$HIGH_BOUNDARY" \
  --min_timestep_boundary 0 \
  --enable_wandb_log \
  --wandb_project "${WANDB_PROJECT}" \
  --initialize_model_on_cpu

WANDB_NAME="${RUN_PREFIX}-${RUN_KIND}-low-noise" accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_DIR" \
  --dataset_metadata_path "$METADATA_PATH" \
  --data_file_keys "$DATA_FILE_KEYS" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "[[\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"],\"../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth\",\"../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth\"]" \
  --learning_rate 1e-4 \
  --num_epochs "$NUM_EPOCHS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$LOW_OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "$LORA_RANK" \
  --extra_inputs "$EXTRA_INPUTS" \
  "${SCHEDULER_ARGS[@]}" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary "$LOW_BOUNDARY" \
  --enable_wandb_log \
  --wandb_project "${WANDB_PROJECT}" \
  --initialize_model_on_cpu

if [ ! -f "$HIGH_EXPORT" ]; then
  echo "Missing high-noise LoRA checkpoint: $HIGH_EXPORT"
  exit 1
fi

if [ ! -f "$LOW_EXPORT" ]; then
  echo "Missing low-noise LoRA checkpoint: $LOW_EXPORT"
  exit 1
fi

cp "$HIGH_EXPORT" "$OUTPUT_DIR/backbone_high_noise.safetensors"
cp "$LOW_EXPORT" "$OUTPUT_DIR/backbone_low_noise.safetensors"

echo "StoryMem LoRA folder is ready: $OUTPUT_DIR"
echo "  $OUTPUT_DIR/backbone_high_noise.safetensors"
echo "  $OUTPUT_DIR/backbone_low_noise.safetensors"
