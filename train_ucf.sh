DATASET_ROOT="data/ucf101_storymem"
LORA_ROOT="../StoryMem/examples/motion_loras/ucf101"
LOG_ROOT="logs"

mkdir -p "$LOG_ROOT"

for dataset_dir in "$DATASET_ROOT"/*; do
  sample_name="$(basename "$dataset_dir")"

  bash examples/wanvideo/model_training/lora/train_storymem_mi2v_cut_false_folder.sh \
    "$dataset_dir" \
    "$LORA_ROOT/$sample_name" \
    > "$LOG_ROOT/${sample_name}.log" \
    2> "$LOG_ROOT/${sample_name}_err.log"
done