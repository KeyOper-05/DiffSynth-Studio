#!/usr/bin/env bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: bash $0 <dataset_dir> <storymem_lora_output_dir>"
  echo "Example: bash $0 data/6am ../StoryMem/examples/motion_loras/6am"
  exit 1
fi

DATASET_DIR="$1"
OUTPUT_DIR="$2"
METADATA_PATH="$DATASET_DIR/metadata.csv"
HIGH_OUTPUT_DIR="$OUTPUT_DIR/diffsynth_high_noise_lora"
LOW_OUTPUT_DIR="$OUTPUT_DIR/diffsynth_low_noise_lora"
HIGH_EXPORT="$HIGH_OUTPUT_DIR/epoch-4.safetensors"
LOW_EXPORT="$LOW_OUTPUT_DIR/epoch-4.safetensors"

if [ ! -f "$METADATA_PATH" ]; then
  echo "Missing metadata file: $METADATA_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_DIR" \
  --dataset_metadata_path "$METADATA_PATH" \
  --data_file_keys "video,memory_images,input_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "[[\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"],\"../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth\",\"../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth\"]" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$HIGH_OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 16 \
  --extra_inputs "memory_images,input_image" \
  --training_scheduler_shift 4.0 \
  --max_timestep_boundary 0.308 \
  --min_timestep_boundary 0 \
  --initialize_model_on_cpu

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET_DIR" \
  --dataset_metadata_path "$METADATA_PATH" \
  --data_file_keys "video,memory_images,input_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "[[\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"],\"../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth\",\"../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth\"]" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$LOW_OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 16 \
  --extra_inputs "memory_images,input_image" \
  --training_scheduler_shift 4.0 \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.308 \
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
