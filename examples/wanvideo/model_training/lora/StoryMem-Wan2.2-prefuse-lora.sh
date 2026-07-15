python examples/wanvideo/model_training/lora/prefuse_storymem_lora.py \
  --base_shards "../Wan2.2_Pretrained/i2v/high_noise_model/diffusion_pytorch_model-*.safetensors" \
  --lora_path "../StoryMem_Pretrained/Wan2.2-MI2V-A14B/backbone_high_noise.safetensors" \
  --output_dir "./models/storymem_prefused/Wan2.2-MI2V-A14B/high_noise_model" \
  --lora_alpha 128 \
  --lora_rank 128 \
  --rslora \
  --compute_device npu:0 \
  --compute_dtype bf16

python examples/wanvideo/model_training/lora/prefuse_storymem_lora.py \
  --base_shards "../Wan2.2_Pretrained/i2v/low_noise_model/diffusion_pytorch_model-*.safetensors" \
  --lora_path "../StoryMem_Pretrained/Wan2.2-MI2V-A14B/backbone_low_noise.safetensors" \
  --output_dir "./models/storymem_prefused/Wan2.2-MI2V-A14B/low_noise_model" \
  --lora_alpha 128 \
  --lora_rank 128 \
  --rslora \
  --compute_device npu:0 \
  --compute_dtype bf16
