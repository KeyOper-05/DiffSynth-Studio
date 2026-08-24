mkdir ../loras/$1_mi2v
cp models/train/$1_mi2v_cut_false_high_noise_lora/epoch-4.safetensors ../loras/$1_mi2v/backbone_high_noise.safetensors
cp models/train/$1_mi2v_cut_false_low_noise_lora/epoch-4.safetensors ../loras/$1_mi2v/backbone_low_noise.safetensors