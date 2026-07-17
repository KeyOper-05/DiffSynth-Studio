export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

if [ -z "$1" ]; then
  echo "Usage: bash $0 <action_name>"
  exit 1
fi
ACTION_NAME="$1"
TRAINING_SCHEDULER_SHIFT="${TRAINING_SCHEDULER_SHIFT:-4.0}"
TIMESTEP_BOUNDARY="${TIMESTEP_BOUNDARY:-0.308}"

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/${ACTION_NAME} \
  --dataset_metadata_path data/${ACTION_NAME}/metadata.csv \
  --data_file_keys "video,memory_images,input_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "[[\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"],\"../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth\",\"../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth\"]" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/${ACTION_NAME}_mi2v_cut_false_high_noise_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 16 \
  --extra_inputs "memory_images,input_image" \
  --training_scheduler_shift "${TRAINING_SCHEDULER_SHIFT}" \
  --max_timestep_boundary "${TIMESTEP_BOUNDARY}" \
  --min_timestep_boundary 0 \
  --initialize_model_on_cpu
# StoryMem M2V boundary is timestep 900. With training scheduler shift=4.0,
# high-noise t >= 900 corresponds to index fraction [0, 0.308).


accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/${ACTION_NAME} \
  --dataset_metadata_path data/${ACTION_NAME}/metadata.csv \
  --data_file_keys "video,memory_images,input_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "[[\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"],\"../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth\",\"../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth\"]" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/${ACTION_NAME}_mi2v_cut_false_low_noise_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 16 \
  --extra_inputs "memory_images,input_image" \
  --training_scheduler_shift "${TRAINING_SCHEDULER_SHIFT}" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary "${TIMESTEP_BOUNDARY}" \
  --initialize_model_on_cpu
# Low-noise t < 900 corresponds to index fraction [0.308, 1).
