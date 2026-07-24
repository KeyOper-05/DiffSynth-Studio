# StoryMem 训练框架 Step-by-Step 用法

本文只覆盖 `DiffSynth-Studio` 中现有 StoryMem 训练链路：从训练样本准备，到训练 MI2V cut=false LoRA，再到导出 StoryMem 可直接加载的 LoRA 文件夹。本文不覆盖 StoryMem inference。

## 0. 进入 DiffSynth-Studio

所有命令默认从 `DiffSynth-Studio` 目录执行：

```bash
cd DiffSynth-Studio
```

确认依赖：

```bash
pip install -e .
pip install accelerate modelscope huggingface_hub
ffmpeg -version
```

训练脚本默认使用：

```text
examples/wanvideo/model_training/full/accelerate_config_14B.yaml
```

如果机器卡数、分布式配置或混合精度设置不同，先修改这个 accelerate 配置文件。

## 1. 准备源视频

把每个要定制的动作视频放到 `DiffSynth-Studio/data/` 下，例如：

```text
data/6am.mp4
data/flip.mp4
data/party.mp4
```

每个动作最终会对应一个训练样本目录：

```text
data/6am/
data/flip/
data/party/
```

这些目录里会生成 `metadata.csv`、训练视频、memory 图片和 `input_image`。

## 2. 制作 MI2V cut=false 训练样本

推荐直接调用 Python 数据制作脚本：

```bash
python examples/wanvideo/model_training/scripts/make_storymem_cut_false_dataset.py \
  --video data/6am.mp4 \
  --prev-end "00:00:00.000" \
  --start "00:00:00.000" \
  --end "00:00:04.000" \
  --prompt "A man wakes up in bed, shifts restlessly, then grabs a pillow, rolls onto his side, and pulls it over his head." \
  --output data/6am \
  --num-frames 49 \
  --fps 16 \
  --width 832 \
  --height 480 \
  --quality 8 \
  --overwrite
```

也可以运行当前示例 shell：

```bash
bash examples/wanvideo/model_training/scripts/make_storymem_cut_false_dataset.sh
```

这个示例 shell 目前是直接调用形式；如果要换动作，需要直接修改里面传给 Python 脚本的参数。

参数含义：

- `--video`：源视频路径。
- `--prev-end`：上一 shot 最后一帧的时间。
- `--start`：当前 shot 开始时间。
- `--end`：当前 shot 结束时间。
- `--prompt`：当前 shot 的训练文本描述。
- `--output`：输出训练样本目录，通常是 `data/<action_name>`。
- `--num-frames`：训练视频帧数，当前训练脚本默认用 `49`。
- `--fps`：输出训练视频 fps，当前 StoryMem/Wan 设置使用 `16`。
- `--width`、`--height`：训练分辨率，当前脚本使用 `832x480`。

如果不显式传 `--memory-frame` 或 `--memory-time`，数据脚本会默认采样三张 memory 图片：

```text
源视频首帧
当前采样段前 1 帧
当前采样段中间任意 1 帧
```

生成后目录结构大致是：

```text
data/6am/
  metadata.csv
  videos/sample_000000.mp4
  input_images/sample_000000/previous_last_frame.png
  memory/sample_000000/memory_000.png
  memory/sample_000000/memory_001.png
  memory/sample_000000/memory_002.png
```

`metadata.csv` 字段为：

```csv
video,prompt,memory_images,input_image,sample_mode
```

训练时会读取：

```bash
--data_file_keys "video,memory_images,input_image"
--extra_inputs "memory_images,input_image"
```

## 3. 检查训练样本目录

训练前至少确认：

```bash
test -f data/6am/metadata.csv
test -d data/6am/videos
test -d data/6am/memory
test -d data/6am/input_images
```

可以打开 `metadata.csv` 看一眼，确认 `memory_images` 是 JSON list 字符串，路径都相对 `data/6am`。

## 4. 推荐训练方式：直接输出 StoryMem LoRA 文件夹

如果目标是给 StoryMem inference 使用，推荐使用包装脚本：

```bash
bash examples/wanvideo/model_training/lora/train_storymem_mi2v_cut_false_folder.sh \
  data/6am \
  ../StoryMem/examples/motion_loras/6am
```

第一个参数是训练样本目录。第二个参数是 StoryMem LoRA 输出目录。

这个脚本会连续训练两次：

- high-noise LoRA
- low-noise LoRA

训练参数固定为当前已核对设置：

```text
num_frames = 49
fps = 16
height = 480
width = 832
lora_rank = 16
training_scheduler_shift = 4.0
timestep_boundary = 0.308
```

high-noise 训练范围：

```bash
--min_timestep_boundary 0
--max_timestep_boundary 0.308
```

low-noise 训练范围：

```bash
--min_timestep_boundary 0.308
--max_timestep_boundary 1
```

训练完成后输出：

```text
../StoryMem/examples/motion_loras/6am/
  diffsynth_high_noise_lora/epoch-4.safetensors
  diffsynth_low_noise_lora/epoch-4.safetensors
  backbone_high_noise.safetensors
  backbone_low_noise.safetensors
```

其中这两个文件是 StoryMem 推理端直接期待的文件名：

```text
backbone_high_noise.safetensors
backbone_low_noise.safetensors
```

## 5. 旧训练入口：按 action_name 输出到 DiffSynth models/train

如果只想使用原始训练入口：

```bash
bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-MI2V-cut-false-A14B.sh 6am
```

这个脚本固定读取：

```text
data/6am/metadata.csv
```

训练输出为：

```text
models/train/6am_mi2v_cut_false_high_noise_lora/epoch-4.safetensors
models/train/6am_mi2v_cut_false_low_noise_lora/epoch-4.safetensors
```

如果要给 StoryMem 使用，需要手动整理：

```bash
mkdir -p ../StoryMem/examples/motion_loras/6am

cp models/train/6am_mi2v_cut_false_high_noise_lora/epoch-4.safetensors \
  ../StoryMem/examples/motion_loras/6am/backbone_high_noise.safetensors

cp models/train/6am_mi2v_cut_false_low_noise_lora/epoch-4.safetensors \
  ../StoryMem/examples/motion_loras/6am/backbone_low_noise.safetensors
```

这一步就是 `train_storymem_mi2v_cut_false_folder.sh` 自动做的事情。

## 6. 批量训练多个动作

对每个动作重复数据制作和训练即可。例如：

```bash
python examples/wanvideo/model_training/scripts/make_storymem_cut_false_dataset.py \
  --video data/flip.mp4 \
  --prev-end "00:00:00.000" \
  --start "00:00:00.000" \
  --end "00:00:04.000" \
  --prompt "A person performs a flip." \
  --output data/flip \
  --num-frames 49 \
  --fps 16 \
  --width 832 \
  --height 480 \
  --quality 8 \
  --overwrite

bash examples/wanvideo/model_training/lora/train_storymem_mi2v_cut_false_folder.sh \
  data/flip \
  ../StoryMem/examples/motion_loras/flip
```

然后对 `6am`、`party` 等其他动作重复同样流程。

仓库里仍保留了一个旧批量训练示例：

```bash
bash test_storymem.sh
```

它会依次调用旧 action 训练脚本训练：

```text
flip
6am
party
```

但它只负责训练到 `models/train/...`，不会自动导出成 StoryMem 的 `backbone_high_noise.safetensors` / `backbone_low_noise.safetensors` 文件夹格式。

## 7. 训练完成后的最小核对

如果使用推荐包装脚本，训练完成后检查：

```bash
test -f ../StoryMem/examples/motion_loras/6am/backbone_high_noise.safetensors
test -f ../StoryMem/examples/motion_loras/6am/backbone_low_noise.safetensors
```

可选检查 LoRA key：

```bash
python examples/wanvideo/model_training/lora/inspect_storymem_lora_keys.py \
  --safetensors ../StoryMem/examples/motion_loras/6am/backbone_high_noise.safetensors \
  --strip-wrapper-prefix

python examples/wanvideo/model_training/lora/inspect_storymem_lora_keys.py \
  --safetensors ../StoryMem/examples/motion_loras/6am/backbone_low_noise.safetensors \
  --strip-wrapper-prefix
```

## 8. 当前训练框架边界

当前 DiffSynth 侧训练框架负责：

- 制作 StoryMem MI2V cut=false 训练样本。
- 读取 `video,memory_images,input_image`。
- 分 high-noise / low-noise 两段训练 LoRA。
- 输出 StoryMem 可接受的 LoRA 文件夹格式。

当前 DiffSynth 侧训练框架不负责：

- 生成 StoryMem inference 的 `shot_loras_json`。
- 启动 StoryMem inference。
- 自动把一个 story 文件拆成多个 action 训练任务。
- 自动选择最新 epoch。

如果只做 train 阶段，推荐最终交付物就是：

```text
../StoryMem/examples/motion_loras/<action_name>/
  backbone_high_noise.safetensors
  backbone_low_noise.safetensors
```
