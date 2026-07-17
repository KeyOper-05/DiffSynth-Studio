export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

# Model paths are configured in StoryMem-Wan2.2-MI2V-cut-false-A14B.sh via --model_paths.

bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-MI2V-cut-false-A14B.sh flip > flip.log 2> flip_err.log

bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-MI2V-cut-false-A14B.sh 6am > 6am.log 2> 6am_err.log

bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-MI2V-cut-false-A14B.sh party > party.log 2> party_err.log
