export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

export STORYMEM_HIGH_NOISE_MODEL="../Wan2.2_Pretrained/i2v/high_noise_model"
export STORYMEM_LOW_NOISE_MODEL="../Wan2.2_Pretrained/i2v/low_noise_model"

export STORYMEM_T5_MODEL="../Wan2.2_Pretrained/i2v/models_t5_umt5-xxl-enc-bf16.pth"
export STORYMEM_VAE_MODEL="../Wan2.2_Pretrained/i2v/Wan2.1_VAE.pth"

bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-MI2V-cut-false-A14B.sh flip > flip.log 2> flip_err.log

bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-MI2V-cut-false-A14B.sh 6am > 6am.log 2> 6am_err.log

bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-MI2V-cut-false-A14B.sh party > party.log 2> party_err.log
