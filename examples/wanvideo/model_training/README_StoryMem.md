# StoryMem 训练启动说明

这份说明对应当前仓库中的 DiffSynth 版 StoryMem 训练脚本：

- 训练入口：`examples/wanvideo/model_training/lora/StoryMem-Wan2.2-T2V-A14B.sh`
- cut=False MI2V 训练入口：`examples/wanvideo/model_training/lora/StoryMem-Wan2.2-MI2V-cut-false-A14B.sh`
- 通用训练代码：`examples/wanvideo/model_training/train.py`
- 单样本数据制作工具：`examples/wanvideo/model_training/scripts/make_storymem_dataset.py`
- cut=False MI2V 单视频拆分工具：`examples/wanvideo/model_training/scripts/make_storymem_cut_false_dataset.py`

当前脚本会按 Wan2.2 A14B 的 high-noise / low-noise 两个 DiT 分段分别训练 LoRA，因此一次启动会连续跑两次 `accelerate launch`，输出两个 LoRA 目录。

## 1. 环境准备

在 `DiffSynth-Studio` 目录下安装依赖：

```bash
cd DiffSynth-Studio
pip install -e .
pip install accelerate modelscope huggingface_hub
```

如果需要用数据制作脚本从视频中切片，还需要系统里能找到 `ffmpeg`：

```bash
ffmpeg -version
```

多卡训练使用 `accelerate`，脚本默认读取：

```text
examples/wanvideo/model_training/full/accelerate_config_14B.yaml
```

如需修改机器数、进程数、混合精度等配置，直接改这个 yaml 或换成自己的 accelerate config。

## 2. 数据格式

StoryMem 训练数据目录默认是：

```text
data/storymem_single_shot/
```

其中 `metadata.csv` 至少包含三列：

```csv
video,prompt,memory_images
videos/sample_000000.mp4,"A character walks into the room.","[""memory/sample_000000/memory_000.png""]"
```

字段含义：

- `video`：当前要训练的单 shot 视频，路径相对 `--dataset_base_path`。
- `prompt`：当前 shot 的文本描述。
- `memory_images`：JSON list 字符串，里面是 memory 图片路径，路径同样相对 `--dataset_base_path`。

训练时脚本会用：

```bash
--data_file_keys "video,memory_images"
--extra_inputs "memory_images"
```

所以 `memory_images` 不能为空，否则 `train.py` 会直接报错。

## 3. 快速制作一个训练样本

可以用当前仓库提供的脚本，从一段源视频中截取一个 5 秒左右的训练样本，并自动生成 memory 图片和 `metadata.csv`：

```bash
python examples/wanvideo/model_training/scripts/make_storymem_dataset.py \
  --video /path/to/source.mp4 \
  --start 00:01:23.5 \
  --end 00:01:28.5 \
  --prompt "A character walks into the room." \
  --output data/storymem_single_shot \
  --num-frames 49 \
  --fps 16 \
  --width 832 \
  --height 480 \
  --memory-time 00:01:20.0
```

说明：

- `--num-frames` 必须满足 `4n+1`，例如 `49` 或 `81`。
- 不传 `--memory-time` 或 `--memory-frame` 时，脚本会默认取输出视频第一帧作为 memory。
- 可以多次传 `--memory-time` 或 `--memory-frame`，生成多张 memory 图片。
- 重复追加样本时，脚本会继续向同一个 `metadata.csv` 追加行。

## 4. cut=False MI2V 数据和训练

`cut=False --mi2v` 使用独立的数据制备脚本，从同一个连续视频里拆出上一 shot 的 last frame 和当前 shot 的训练帧：

```bash
python examples/wanvideo/model_training/scripts/make_storymem_cut_false_dataset.py \
  --video /path/to/full_or_adjacent_shots.mp4 \
  --prev-end 00:00:05.000 \
  --start 00:00:05.000 \
  --end 00:00:10.000 \
  --prompt "The character continues walking into the room." \
  --output data/storymem_cut_false_mi2v \
  --num-frames 49 \
  --fps 16 \
  --width 832 \
  --height 480
```

也可以用配套 shell 封装：

```bash
bash examples/wanvideo/model_training/scripts/make_storymem_cut_false_dataset.sh
```

使用这个 `.sh` 时，直接编辑脚本顶部的 `VIDEO`、`PREV_END`、`START`、`END`、`PROMPT`、`OUTPUT` 等变量即可。

生成的 `metadata.csv` 包含：

```csv
video,prompt,memory_images,input_image,sample_mode
```

其中 `input_image` 是上一 shot 的 `previous_last_frame.png`，训练视频的第 0 帧也会使用同一张图；第 1 帧开始才是当前 shot 的均匀抽帧。脚本会读取源视频 fps，并自动从 `--prev-end` 的次帧开始抽当前 shot，避免 `--prev-end == --start` 时重复包含上一 shot last frame。默认不传 `--memory-time` 或 `--memory-frame` 时，`memory_images` 也会指向这张 previous last frame。

对应训练入口：

```bash
cd DiffSynth-Studio
bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-MI2V-cut-false-A14B.sh
```

这个脚本默认读取：

```text
data/storymem_cut_false_mi2v/metadata.csv
```

并使用：

```bash
--data_file_keys "video,memory_images,input_image"
--extra_inputs "memory_images,input_image"
```

这条路径只覆盖 StoryMem 的 `cut=False --mi2v` 情形：`input_image` 作为上一 shot last frame condition，`memory_images` 继续走 StoryMem memory condition。它不实现 `--mm2v`，也不使用 motion frames 作为前 5 帧条件。

训练 loss 会跳过 StoryMem memory 前缀；当样本包含 `input_image` 时，也会跳过视频 latent 的第 0 帧，只对当前 shot 后续生成帧计算 loss。

## 5. 确认 ModelConfig 适配

DiffSynth 加载模型时不是只看文件路径。`ModelConfig` 会先下载或定位 checkpoint，然后 `auto_load_model` 通过 checkpoint 的结构 hash 去 `diffsynth/configs/model_configs.py` 的 `MODEL_CONFIGS` 里查模型结构。如果没有匹配项，训练会在加载模型时报错：

```text
Cannot detect the model type. File: ...
```

所以启动 StoryMem 训练前，需要确认 high-noise / low-noise DiT checkpoint 有适配的 `ModelConfig`。

先计算 checkpoint 的结构 hash：

```bash
python - <<'PY'
from diffsynth.core import hash_model_file

paths = [
    "/path/to/high_noise_model/diffusion_pytorch_model-00001-of-000xx.safetensors",
    "/path/to/low_noise_model/diffusion_pytorch_model-00001-of-000xx.safetensors",
]

for path in paths:
    print(path)
    print(hash_model_file(path))
PY
```

然后在 `diffsynth/configs/model_configs.py` 中搜索这个 hash：

```bash
rg "这里替换成算出来的hash" diffsynth/configs/model_configs.py
```

如果 hash 已经存在，就不需要新增配置。比如当前仓库已经有 Wan2.2 I2V A14B 的配置：

```python
{
    # Example: ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
    "model_hash": "5b013604280dd715f8457c6ed6d6a626",
    "model_name": "wan_video_dit",
    "model_class": "diffsynth.models.wan_video_dit.WanModel",
    "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'require_clip_embedding': False}
}
```

如果你的 StoryMem / MI2V DiT checkpoint 算出来是新的 hash，但结构仍然是 Wan2.2 I2V A14B 这类 VAE-conditioned DiT，就需要在 `wan_series = [` 里新增一条对应配置。模板如下，把 `model_hash` 换成实际算出来的 hash：

```python
{
    # Example: ModelConfig(model_id="Your-Org/StoryMem-MI2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
    "model_hash": "替换成你的StoryMem或MI2V-DiT结构hash",
    "model_name": "wan_video_dit",
    "model_class": "diffsynth.models.wan_video_dit.WanModel",
    "extra_kwargs": {
        'has_image_input': False,
        'patch_size': [1, 2, 2],
        'in_dim': 36,
        'dim': 5120,
        'ffn_dim': 13824,
        'freq_dim': 256,
        'text_dim': 4096,
        'out_dim': 16,
        'num_heads': 40,
        'num_layers': 40,
        'eps': 1e-06,
        'require_clip_embedding': False,
    }
}
```

high-noise 和 low-noise 如果只有权重数值不同、参数名和 shape 相同，结构 hash 会相同，只需要一条 `ModelConfig`。如果两者 hash 不同，就分别加两条。

## 6. 配置模型路径

启动训练前必须设置 high-noise 和 low-noise 两个 DiT 模型入口。这里需要指向可被 `ModelConfig` 识别的 DiT checkpoint，不是最终训练出来的 LoRA。可以写 Hugging Face / ModelScope 的 `model_id:pattern`，也可以写本地 checkpoint 路径。

以 Wan2.2 I2V A14B 作为兼容的 VAE-conditioned DiT 初始化为例：

```bash
export STORYMEM_HIGH_NOISE_MODEL="Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors"
export STORYMEM_LOW_NOISE_MODEL="Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors"
```

脚本还会使用 T5 和 VAE，默认值如下，通常不用改：

```bash
export STORYMEM_T5_MODEL="Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth"
export STORYMEM_VAE_MODEL="Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth"
```

如果模型已经下载到本地，也可以改成你的本地路径，例如：

```bash
export STORYMEM_HIGH_NOISE_MODEL="/path/to/high_noise_model/diffusion_pytorch_model-00001-of-000xx.safetensors"
export STORYMEM_LOW_NOISE_MODEL="/path/to/low_noise_model/diffusion_pytorch_model-00001-of-000xx.safetensors"
```

注意：StoryMem memory conditioning 需要 DiT 支持 VAE conditioning channels。不要用普通 T2V DiT 直接替代 high/low 模型，否则会在 `StoryMem memory conditioning requires a Wan DiT with VAE conditioning channels` 处报错。即使路径能解析，仍然要先满足第 5 节的 `ModelConfig` hash 匹配。

## 7. 启动训练

最小启动命令：

```bash
cd DiffSynth-Studio

export STORYMEM_HIGH_NOISE_MODEL="Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors"
export STORYMEM_LOW_NOISE_MODEL="Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors"

bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-T2V-A14B.sh
```

默认读取：

```text
DATASET_BASE=data/storymem_single_shot
METADATA_PATH=data/storymem_single_shot/metadata.csv
```

默认输出：

```text
./models/train/StoryMem-Wan2.2-T2V-A14B_high_noise_lora
./models/train/StoryMem-Wan2.2-T2V-A14B_low_noise_lora
```

如果数据和输出目录不同，可以通过环境变量覆盖：

```bash
DATASET_BASE=/path/to/storymem_dataset \
METADATA_PATH=/path/to/storymem_dataset/metadata.csv \
OUTPUT_PATH_HIGH=./models/train/my_storymem_high_lora \
OUTPUT_PATH_LOW=./models/train/my_storymem_low_lora \
bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-T2V-A14B.sh
```

## 8. 常用训练参数

当前 shell 脚本支持用环境变量覆盖这些常用参数：

```bash
HEIGHT=480
WIDTH=832
NUM_FRAMES=49
DATASET_REPEAT=100
LEARNING_RATE=1e-4
NUM_EPOCHS=5
LORA_RANK=32
```

示例：

```bash
HEIGHT=480 \
WIDTH=832 \
NUM_FRAMES=81 \
DATASET_REPEAT=50 \
LEARNING_RATE=5e-5 \
NUM_EPOCHS=10 \
LORA_RANK=64 \
bash examples/wanvideo/model_training/lora/StoryMem-Wan2.2-T2V-A14B.sh
```

## 9. 脚本实际做了什么

`StoryMem-Wan2.2-T2V-A14B.sh` 组装了一组公共参数，然后分两段启动训练：

1. high-noise LoRA：

```bash
--model_id_with_origin_paths "${STORYMEM_HIGH_NOISE_MODEL},${STORYMEM_T5_MODEL},${STORYMEM_VAE_MODEL}"
--output_path "./models/train/StoryMem-Wan2.2-T2V-A14B_high_noise_lora"
--max_timestep_boundary 0.1
--min_timestep_boundary 0
```

2. low-noise LoRA：

```bash
--model_id_with_origin_paths "${STORYMEM_LOW_NOISE_MODEL},${STORYMEM_T5_MODEL},${STORYMEM_VAE_MODEL}"
--output_path "./models/train/StoryMem-Wan2.2-T2V-A14B_low_noise_lora"
--max_timestep_boundary 1
--min_timestep_boundary 0.1
```

这里和原 StoryMem 的 `boundary = 0.900` 对齐。DiffSynth 训练时的 Wan timestep 按高到低排列，所以 `t >= 900` 的 high-noise 区间对应 index fraction `[0, 0.1)`，`t < 900` 的 low-noise 区间对应 `[0.1, 1)`。

公共 LoRA 配置为：

```bash
--lora_base_model "dit"
--lora_target_modules "q,k,v,o,ffn.0,ffn.2"
--lora_rank 32
--remove_prefix_in_ckpt "pipe.dit."
```

StoryMem 的关键差异是额外传入：

```bash
--data_file_keys "video,memory_images"
--extra_inputs "memory_images"
```

`train.py` 会把 `memory_images` 读成 PIL 图片列表，交给 `WanVideoUnit_StoryMemMemoryEmbedder` 编码成 memory latents，并在 DiT 前向时使用 StoryMem 的 temporal RoPE 位置。

## 10. 常见问题

`STORYMEM_HIGH_NOISE_MODEL` 或 `STORYMEM_LOW_NOISE_MODEL` 未设置：

脚本开头会强制检查这两个变量。按第 6 节先 `export` 再启动。

`Cannot detect the model type`：

checkpoint 的结构 hash 没有命中 `diffsynth/configs/model_configs.py`。按第 5 节计算 hash，并为 StoryMem / MI2V DiT 增加匹配的 `ModelConfig`。

`memory_images` 报错为空或格式不对：

检查 `metadata.csv` 中 `memory_images` 是否是合法 JSON list 字符串，例如：

```csv
"[""memory/sample_000000/memory_000.png"", ""memory/sample_000000/memory_001.png""]"
```

普通 T2V DiT 报 VAE conditioning 错误：

StoryMem 需要能接收 VAE 条件通道的 DiT。请使用 Wan2.2 I2V / MI2V 这类兼容 checkpoint，或者使用已经为 StoryMem/MI2V 改好的高低噪 DiT。

显存不够：

优先降低 `HEIGHT`、`WIDTH`、`NUM_FRAMES` 或 `LORA_RANK`。如果只做单卡低显存实验，也可以直接调用 `train.py` 并加上 DiffSynth 训练框架支持的 `--enable_model_cpu_offload`，但当前封装脚本没有单独暴露这个变量。
