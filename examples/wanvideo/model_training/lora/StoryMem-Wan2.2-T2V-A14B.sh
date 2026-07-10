#!/usr/bin/env bash
# Shebang: asks env to locate bash and execute this script with it.
set -e
# Bash option: exit immediately if any command returns a non-zero status.

# Parameter expansion ${VAR:=default}: high-noise DiT checkpoint/model entry.
: "${STORYMEM_HIGH_NOISE_MODEL:=Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors}"
# Parameter expansion ${VAR:=default}: low-noise DiT checkpoint/model entry.
: "${STORYMEM_LOW_NOISE_MODEL:=Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors}"
# Parameter expansion ${VAR:=default}: default shared Wan T5 encoder model entry.
: "${STORYMEM_T5_MODEL:=Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth}"
# Parameter expansion ${VAR:=default}: default shared Wan VAE model entry.
: "${STORYMEM_VAE_MODEL:=Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth}"

# Parameter expansion ${VAR:-default}: dataset root containing videos, memory images, metadata.csv.
DATASET_BASE="${DATASET_BASE:-data/storymem_single_shot}"
# Default metadata CSV path; must include video, prompt, and memory_images columns.
METADATA_PATH="${METADATA_PATH:-${DATASET_BASE}/metadata.csv}"

# Bash array: common flags shared by both high-noise and low-noise training commands.
COMMON_ARGS=(
  # train.py argument: root prepended by ToAbsolutePath for relative data paths.
  --dataset_base_path "${DATASET_BASE}"
  # train.py argument: CSV/JSON/JSONL metadata consumed by UnifiedDataset.
  --dataset_metadata_path "${METADATA_PATH}"
  # UnifiedDataset processes both video and memory_images through operators.
  --data_file_keys "video,memory_images"
  # add_video_size_config argument: resize/crop training video height.
  --height "${HEIGHT:-480}"
  # add_video_size_config argument: resize/crop training video width.
  --width "${WIDTH:-832}"
  # add_video_size_config argument: current-shot pixel frame count, usually 4n+1.
  --num_frames "${NUM_FRAMES:-49}"
  # add_general_config argument: logical repeat count for dataset epochs.
  --dataset_repeat "${DATASET_REPEAT:-100}"
  # Optimizer learning rate for LoRA training.
  --learning_rate "${LEARNING_RATE:-1e-4}"
  # Number of training epochs.
  --num_epochs "${NUM_EPOCHS:-5}"
  # ModelLogger strips this prefix when saving LoRA/full checkpoints.
  --remove_prefix_in_ckpt "pipe.dit."
  # switch_pipe_to_training_mode target: attach LoRA to pipe.dit.
  --lora_base_model "dit"
  # LoRA module name filters inside Wan DiT blocks.
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
  # LoRA rank hyperparameter.
  --lora_rank "${LORA_RANK:-32}"
  # WanTrainingModule.parse_extra_inputs passes loaded memory images into pipeline inputs_shared.
  --extra_inputs "memory_images"
)

# accelerate launch runs distributed training; this command trains high-noise LoRA.
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py   "${COMMON_ARGS[@]}"   --model_id_with_origin_paths "${STORYMEM_HIGH_NOISE_MODEL},${STORYMEM_T5_MODEL},${STORYMEM_VAE_MODEL}"   --output_path "${OUTPUT_PATH_HIGH:-./models/train/StoryMem-Wan2.2-T2V-A14B_high_noise_lora}"   --max_timestep_boundary 0.417   --min_timestep_boundary 0
# High-noise command details: model_id_with_origin_paths is parsed by parse_model_configs; output_path stores high LoRA; timestep range is [0, 0.417].

# accelerate launch runs distributed training; this command trains low-noise LoRA.
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py   "${COMMON_ARGS[@]}"   --model_id_with_origin_paths "${STORYMEM_LOW_NOISE_MODEL},${STORYMEM_T5_MODEL},${STORYMEM_VAE_MODEL}"   --output_path "${OUTPUT_PATH_LOW:-./models/train/StoryMem-Wan2.2-T2V-A14B_low_noise_lora}"   --max_timestep_boundary 1   --min_timestep_boundary 0.417
# Low-noise command details: model_id_with_origin_paths loads low DiT; output_path stores low LoRA; timestep range is [0.417, 1].
