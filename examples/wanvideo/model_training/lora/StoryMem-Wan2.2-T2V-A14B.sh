STORYMEM_HIGH_NOISE_MODEL_PATHS="${STORYMEM_HIGH_NOISE_MODEL_PATHS:-[\"../Wan2.2_Pretrained/i2v/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",\"../Wan2.2_Pretrained/i2v/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",\"../Wan2.2_Pretrained/i2v/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",\"../Wan2.2_Pretrained/i2v/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",\"../Wan2.2_Pretrained/i2v/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",\"../Wan2.2_Pretrained/i2v/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"]}"
STORYMEM_LOW_NOISE_MODEL_PATHS="${STORYMEM_LOW_NOISE_MODEL_PATHS:-[\"../Wan2.2_Pretrained/i2v/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",\"../Wan2.2_Pretrained/i2v/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",\"../Wan2.2_Pretrained/i2v/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",\"../Wan2.2_Pretrained/i2v/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",\"../Wan2.2_Pretrained/i2v/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",\"../Wan2.2_Pretrained/i2v/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"]}"
STORYMEM_T5_MODEL_PATH="${STORYMEM_T5_MODEL_PATH:-\"../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth\"}"
STORYMEM_VAE_MODEL_PATH="${STORYMEM_VAE_MODEL_PATH:-\"../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth\"}"

STORYMEM_PRESET_LORA_DIR="${STORYMEM_PRESET_LORA_DIR-../StoryMem_Pretrained/Wan2.2-MI2V-A14B}"
PRESET_HIGH_LORA_ARGS=()
PRESET_LOW_LORA_ARGS=()
if [ -n "${STORYMEM_PRESET_LORA_DIR}" ]; then
  STORYMEM_PRESET_HIGH_LORA="${STORYMEM_PRESET_HIGH_LORA:-${STORYMEM_PRESET_LORA_DIR}/backbone_high_noise.safetensors}"
  STORYMEM_PRESET_LOW_LORA="${STORYMEM_PRESET_LOW_LORA:-${STORYMEM_PRESET_LORA_DIR}/backbone_low_noise.safetensors}"
  PRESET_HIGH_LORA_ARGS=(--preset_lora_path "${STORYMEM_PRESET_HIGH_LORA}" --preset_lora_model "dit")
  PRESET_LOW_LORA_ARGS=(--preset_lora_path "${STORYMEM_PRESET_LOW_LORA}" --preset_lora_model "dit")
fi

DATASET_BASE="${DATASET_BASE:-data/storymem_single_shot}"
METADATA_PATH="${METADATA_PATH:-${DATASET_BASE}/metadata.csv}"

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_BASE}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --data_file_keys "video,memory_images" \
  --height "${HEIGHT:-480}" \
  --width "${WIDTH:-832}" \
  --num_frames "${NUM_FRAMES:-49}" \
  --dataset_repeat "${DATASET_REPEAT:-100}" \
  --model_paths "[${STORYMEM_HIGH_NOISE_MODEL_PATHS},${STORYMEM_T5_MODEL_PATH},${STORYMEM_VAE_MODEL_PATH}]" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --num_epochs "${NUM_EPOCHS:-5}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH_HIGH:-./models/train/StoryMem-Wan2.2-T2V-A14B_high_noise_lora}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "${LORA_RANK:-32}" \
  "${PRESET_HIGH_LORA_ARGS[@]}" \
  --extra_inputs "memory_images" \
  --max_timestep_boundary 0.417 \
  --min_timestep_boundary 0
# boundary corresponds to timesteps [875, 1000]


accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_BASE}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --data_file_keys "video,memory_images" \
  --height "${HEIGHT:-480}" \
  --width "${WIDTH:-832}" \
  --num_frames "${NUM_FRAMES:-49}" \
  --dataset_repeat "${DATASET_REPEAT:-100}" \
  --model_paths "[${STORYMEM_LOW_NOISE_MODEL_PATHS},${STORYMEM_T5_MODEL_PATH},${STORYMEM_VAE_MODEL_PATH}]" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --num_epochs "${NUM_EPOCHS:-5}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH_LOW:-./models/train/StoryMem-Wan2.2-T2V-A14B_low_noise_lora}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "${LORA_RANK:-32}" \
  "${PRESET_LOW_LORA_ARGS[@]}" \
  --extra_inputs "memory_images" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.417
# boundary corresponds to timesteps [0, 875)
