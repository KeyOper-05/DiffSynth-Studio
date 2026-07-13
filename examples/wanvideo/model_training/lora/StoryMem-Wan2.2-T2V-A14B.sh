accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/storymem_single_shot \
  --dataset_metadata_path data/storymem_single_shot/metadata.csv \
  --data_file_keys "video,memory_images" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "[[\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"],\"../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth\",\"../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth\"]" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/StoryMem-Wan2.2-T2V-A14B_high_noise_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "memory_images" \
  --max_timestep_boundary 0.417 \
  --min_timestep_boundary 0 \
  --debug_checkpoints \
  --debug_checkpoint_trace_after 300 \
  --initialize_model_on_cpu
# boundary corresponds to timesteps [875, 1000]


accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/storymem_single_shot \
  --dataset_metadata_path data/storymem_single_shot/metadata.csv \
  --data_file_keys "video,memory_images" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "[[\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",\"./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"],\"../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth\",\"../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth\"]" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/StoryMem-Wan2.2-T2V-A14B_low_noise_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "memory_images" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.417 \
  --debug_checkpoints \
  --debug_checkpoint_trace_after 300 \
  --initialize_model_on_cpu
# boundary corresponds to timesteps [0, 875)
