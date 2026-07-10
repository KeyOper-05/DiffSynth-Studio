# `diffsynth` Wan Video 开发文档

本文面向需要阅读、修改或复用 Wan Video 相关代码的开发者，覆盖 `diffsynth/pipelines/wan_video.py` 以及 `diffsynth/models/wan_video_*` 中的主要组件。使用层面的模型列表和推理示例见 [Wan 模型详解](../../Model_Details/Wan.md)。

## 代码范围

核心文件：

|文件|职责|
|-|-|
|`diffsynth/pipelines/wan_video.py`|Wan 系列统一 Pipeline；负责模型装配、输入预处理、扩展条件构建、扩散迭代、VAE 解码。|
|`diffsynth/models/wan_video_dit.py`|标准 Wan DiT 主干，包括 patchify、RoPE、self/cross attention、DiT block、输出 head。|
|`diffsynth/models/wan_video_vae.py`|视频 VAE 编解码，支持普通、分块、逐帧编码/解码。|
|`diffsynth/models/wan_video_text_encoder.py`|T5 文本编码器和 HuggingFace tokenizer 包装。|
|`diffsynth/models/wan_video_image_encoder.py`|CLIP/XLM-RoBERTa 图像编码器包装，提供图像条件 token。|
|`diffsynth/models/wan_video_vace.py`|VACE 控制分支，生成注入 DiT block 的控制 hint。|
|`diffsynth/models/wan_video_dit_s2v.py`|Speech-to-Video DiT 变体，处理音频、姿态和 motion 条件。|
|`diffsynth/models/wan_video_mot.py`|Video-As-Prompt/MoT 分支。|
|`diffsynth/models/wan_video_animate_adapter.py`|Animate 姿态和人脸适配器。|
|`diffsynth/models/wan_video_motion_controller.py`|速度/运动 bucket 控制器。|
|`diffsynth/models/longcat_video_dit.py`|LongCat-Video 变体。|
|`diffsynth/models/wav2vec.py`|Wan S2V 音频特征编码器。|

## 数据流

`WanVideoPipeline.__call__` 的数据流可以分为六段：

1. 调度器初始化：`FlowMatchScheduler.set_timesteps(num_inference_steps, denoising_strength, shift)` 生成扩散时间步。
2. 输入字典拆分：正向提示词进入 `inputs_posi`，负向提示词进入 `inputs_nega`，图像、视频、音频、形状、控制条件等进入 `inputs_shared`。
3. `PipelineUnit` 预处理：`self.units` 依次消费并更新三个输入字典，生成 `noise`、`latents`、`context`、`clip_feature`、`y`、`vace_context`、`audio_embeds`、`pose_latents` 等模型输入。
4. 扩散迭代：每个 timestep 调用 `model_fn_wan_video` 预测噪声；当 `cfg_scale != 1.0` 时，分别计算正/负条件或通过 `cfg_merge` 合并 batch，然后执行 classifier-free guidance。
5. 调度器更新：`FlowMatchScheduler.step(noise_pred, timestep, latents)` 更新 latent；若使用首帧融合条件，则恢复 `first_frame_latents`。
6. 后处理与解码：`post_units` 处理 S2V motion latent；`WanVideoVAE.decode` 或 `decode_framewise` 将 latent 转回视频；`vae_output_to_video` 将浮点输出量化为图像帧列表。

主要张量约定：

|名称|形状/类型|含义|
|-|-|-|
|`input_video`|`list[PIL.Image.Image]` 或预处理后 `torch.Tensor[B,3,T,H,W]`|视频到视频或训练目标。|
|`noise`|`torch.Tensor[1,C,T_lat,H_lat,W_lat]`|扩散初始噪声。|
|`latents`|`torch.Tensor[B,C,T_lat,H_lat,W_lat]`|扩散过程中的当前 latent。|
|`input_latents`|`torch.Tensor[B,C,T_lat,H_lat,W_lat]`|由输入视频编码出的干净 latent，训练时用于加噪目标。|
|`context`|`torch.Tensor[B,L,D]`|文本编码器输出。|
|`clip_feature`|`torch.Tensor[B,257或514,1280]`|图像编码器输出，I2V/FLF2V 使用。|
|`y`|`torch.Tensor[B,4+C,T_lat,H_lat,W_lat]` 或控制变体通道|VAE 图像/视频条件，前 4 个通道通常是 mask。|
|`vace_context`|`torch.Tensor[B,2C+64,T_lat,H_lat,W_lat]`|VACE inactive/reactive latent 与 mask latent 拼接后的控制条件。|
|`audio_embeds`|`torch.Tensor`|S2V 音频编码器输出。|
|`memory_size`|`int`|StoryMem 前缀 memory latent 帧数；为 0 时保持标准 Wan RoPE。|

### PipelineUnit 顺序

`WanVideoPipeline.units` 的顺序决定了条件覆盖关系：

|顺序|类|输入到输出|
|-|-|-|
|1|`WanVideoUnit_ShapeChecker`|校正 `height`、`width`、`num_frames`，满足 Pipeline 的整除约束。|
|2|`WanVideoUnit_NoiseInitializer`|根据 VAE latent 尺寸生成 `noise`。|
|3|`WanVideoUnit_PromptEmbedder`|`prompt`/`negative_prompt` -> `context`。|
|4|`WanVideoUnit_S2V`|音频、pose、motion video -> S2V 条件。|
|5|`WanVideoUnit_InputVideoEmbedder`|`input_video` -> `input_latents`，推理时加噪得到 `latents`。|
|6|`WanVideoUnit_ImageEmbedderVAE`|`input_image`/`end_image` -> VAE 条件 `y`。|
|7|`WanVideoUnit_StoryMemMemoryEmbedder`|`memory_images` -> memory latent 前缀、StoryMem mask、`memory_size`。|
|8|`WanVideoUnit_ImageEmbedderCLIP`|`input_image`/`end_image` -> `clip_feature`。|
|9|`WanVideoUnit_ImageEmbedderFused`|Wan2.2 TI2V 首帧 latent 融合到 `latents`。|
|10-12|`WanVideoUnit_FunControl`、`FunReference`、`FunCameraControl`|Fun-Control、参考图、相机控制条件。|
|13|`WanVideoUnit_SpeedControl`|`motion_bucket_id` -> tensor。|
|14|`WanVideoUnit_VACE`|VACE 视频、mask、参考图 -> `vace_context`。|
|15-18|Animate units|拆分 animate 视频，编码 pose、face pixel、inpaint 条件。|
|19|`WanVideoUnit_VAP`|Video-As-Prompt 文本、CLIP、latent 条件。|
|20|`WanVideoUnit_UnifiedSequenceParallel`|标记 USP 并行。|
|21|`WanVideoUnit_TeaCache`|构造 TeaCache 状态。|
|22|`WanVideoUnit_CfgMerger`|可选合并正负条件 batch。|
|23-25|LongCat/WanToDance units|LongCat latent、音乐特征、舞蹈参考图和关键帧条件。|

## 类功能和属性

### `WanVideoPipeline`

签名：

```python
class WanVideoPipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16)
    def enable_usp(self)
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(...),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    )
    @torch.no_grad()
    def __call__(...)
```

功能：

|属性|类型|用途|
|-|-|-|
|`scheduler`|`FlowMatchScheduler`|管理 flow matching 时间步、加噪和 step 更新。|
|`tokenizer`|`HuggingfaceTokenizer`|文本转 token id 和 attention mask。|
|`audio_processor`|`Wav2Vec2Processor`|S2V 音频预处理。|
|`text_encoder`|`WanTextEncoder`|T5 文本编码。|
|`image_encoder`|`WanImageEncoder`|CLIP 图像编码。|
|`dit` / `dit2`|`WanModel` 或变体|主去噪网络；`dit2` 用于 Wan2.2 双 DiT 边界切换。|
|`vae`|`WanVideoVAE`|视频和 latent 的互相转换。|
|`motion_controller`|`WanMotionControllerModel`|运动 bucket 调制时间嵌入。|
|`vace` / `vace2`|`VaceWanModel`|VACE 控制分支。|
|`vap`|`MotWanModel`|Video-As-Prompt 分支。|
|`animate_adapter`|`WanAnimateAdapter`|Animate pose/face adapter。|
|`audio_encoder`|`WanS2VAudioEncoder`|音频特征到 S2V embedding。|
|`units`|`list[PipelineUnit]`|推理前条件构建流水线。|
|`post_units`|`list[PipelineUnit]`|去噪后、VAE 解码前的处理单元。|
|`model_fn`|callable|默认是 `model_fn_wan_video`。|
|`compilable_models`|`list[str]`|可被编译/优化的模型名。|

### PipelineUnit 类

所有 `WanVideoUnit_*` 继承 `PipelineUnit`。普通 unit 通过 `input_params` 声明读取字段，通过 `output_params` 声明写回字段；`take_over=True` 的 unit 直接接收并返回 `inputs_shared, inputs_posi, inputs_nega`。

|类|`process` 签名摘要|用途|输入|输出|
|-|-|-|-|-|
|`WanVideoUnit_ShapeChecker`|`process(pipe, height, width, num_frames)`|校正输入尺寸。|整数尺寸。|`height,width,num_frames`。|
|`WanVideoUnit_NoiseInitializer`|`process(pipe, height, width, num_frames, seed, rand_device, vace_reference_image)`|生成初始噪声；VACE 参考帧会扩展时间长度。|形状、随机种子。|`noise`。|
|`WanVideoUnit_InputVideoEmbedder`|`process(pipe, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image, framewise_decoding)`|将输入视频编码为 latent；推理时按首个 timestep 加噪。|视频、噪声、VAE tiling 参数。|`latents,input_latents`。|
|`WanVideoUnit_PromptEmbedder`|`encode_prompt(pipe, prompt)` / `process(pipe, prompt, positive)`|tokenize 并编码文本。|提示词。|`context`。|
|`WanVideoUnit_ImageEmbedderCLIP`|`process(pipe, input_image, end_image, height, width)`|编码 CLIP 图像条件。|首帧/尾帧图像。|`clip_feature`。|
|`WanVideoUnit_ImageEmbedderVAE`|`process(pipe, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride)`|构造 I2V/FLF2V 的 mask + VAE latent 条件。|首帧/尾帧图像。|`y`。|
|`WanVideoUnit_StoryMemMemoryEmbedder`|`process(pipe, memory_images, noise, latents, input_latents, y, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride)`|将 StoryMem 记忆图像编码为 latent 前缀，并构造负时间 RoPE 所需的 `memory_size`。|memory 图像、训练视频 latent、条件。|更新 `noise,latents,input_latents,y,memory_size`。|
|`WanVideoUnit_ImageEmbedderFused`|`process(pipe, input_image, latents, height, width, tiled, tile_size, tile_stride)`|Wan2.2 TI2V 将首帧 VAE latent 直接写入 `latents`。|首帧图像和 latent。|`latents,fuse_vae_embedding_in_latents,first_frame_latents`。|
|`WanVideoUnit_FunControl`|`process(pipe, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents)`|编码 Fun-Control 视频条件并拼到 `y`。|控制视频。|`clip_feature,y`。|
|`WanVideoUnit_FunReference`|`process(pipe, reference_image, height, width)`|编码参考图 latent 和可选 CLIP 条件。|参考图。|`reference_latents,clip_feature`。|
|`WanVideoUnit_FunCameraControl`|`process(pipe, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride)`|根据相机方向生成 Plucker 控制 latent，并构造首帧条件。|相机参数、首帧。|`control_camera_latents_input,y`。|
|`WanVideoUnit_SpeedControl`|`process(pipe, motion_bucket_id)`|把运动 bucket 转为张量。|整数 bucket。|`motion_bucket_id`。|
|`WanVideoUnit_VACE`|`process(pipe, vace_video, vace_video_mask, vace_reference_image, vace_scale, height, width, num_frames, tiled, tile_size, tile_stride)`|构造 VACE inactive/reactive latent、mask latent 和参考图 latent。|VACE 视频/遮罩/参考图。|`vace_context,vace_scale`。|
|`WanVideoUnit_VAP`|`process(pipe, inputs_shared, inputs_posi, inputs_nega)`|构造 Video-As-Prompt 的文本、CLIP 和 latent 条件。|`vap_video,vap_prompt,negative_vap_prompt`。|`vap_clip_feature,vap_hidden_state,context_vap`。|
|`WanVideoUnit_UnifiedSequenceParallel`|`process(pipe)`|把 pipeline 的 USP 状态写入输入字典。|pipeline 状态。|`use_unified_sequence_parallel`。|
|`WanVideoUnit_TeaCache`|`process(pipe, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id)`|构造 TeaCache。|阈值、模型 ID。|`tea_cache`。|
|`WanVideoUnit_CfgMerger`|`process(pipe, inputs_shared, inputs_posi, inputs_nega)`|把正负条件拼 batch，减少一次 DiT forward。|正负条件字典。|合并后的共享字典。|
|`WanVideoUnit_S2V`|`process_audio` / `process_motion_latents` / `process_pose_cond` / `process(...)`|构造 S2V 音频、motion、pose 条件。|音频、pose 视频、motion 视频。|`audio_embeds,motion_latents,drop_motion_frames,s2v_pose_latents`。|
|`WanVideoPostUnit_S2V`|`process(pipe, latents, motion_latents, drop_motion_frames)`|S2V 解码前把 motion latent 拼回时间维。|去噪 latent 和 motion latent。|`latents`。|
|`WanVideoUnit_AnimateVideoSplit`|`process(pipe, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video)`|训练/输入视频存在时裁剪 animate 条件。|animate 视频。|裁剪后的 animate 视频。|
|`WanVideoUnit_AnimatePoseLatents`|`process(pipe, animate_pose_video, tiled, tile_size, tile_stride)`|VAE 编码姿态视频。|pose 视频。|`pose_latents`。|
|`WanVideoUnit_AnimateFacePixelValues`|`process(pipe, inputs_shared, inputs_posi, inputs_nega)`|构造正/负人脸像素条件。|face 视频。|`face_pixel_values`。|
|`WanVideoUnit_AnimateInpaint`|`get_i2v_mask(...)` / `process(...)`|构造 Animate inpaint 的 mask + latent 条件。|inpaint 视频、mask、参考图。|`y`。|
|`WanVideoUnit_LongCatVideo`|`process(pipe, longcat_video)`|编码 LongCat 条件视频。|LongCat 输入视频。|`longcat_latents`。|
|`WanVideoUnit_WanToDance_ProcessInputs`|`get_music_base_feature(music_path, fps=30)` / `process(...)`|提取音乐特征并设置 global 模型负条件跳层。|音乐路径。|`music_feature` 等。|
|`WanVideoUnit_WanToDance_RefImageEmbedder`|`process(pipe, wantodance_reference_image, num_frames, height, width, tiled, tile_size, tile_stride)`|编码 WanToDance 参考图 CLIP 条件。|参考图。|`wantodance_refimage_feature`。|
|`WanVideoUnit_WanToDance_ImageKeyframesEmbedder`|`process(pipe, wantodance_keyframes, wantodance_keyframes_mask, num_frames, height, width, tiled, tile_size, tile_stride)`|编码关键帧 CLIP 和 VAE 条件。|关键帧与 mask。|`clip_feature,y`。|

### `WanModel`

签名：

```python
class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    )
    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None, enable_wantodance_global=False)
    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor)
    def forward(self, x, timestep, context, clip_feature=None, y=None, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False, **kwargs)
```

关键属性：

|属性|用途|
|-|-|
|`patch_embedding`|`nn.Conv3d`，将 `[B,C,T,H,W]` latent patch 化为 DiT token。|
|`text_embedding`|把 T5 hidden size 映射到 DiT hidden size。|
|`time_embedding` / `time_projection`|将 timestep 转为 AdaLN 调制参数。|
|`blocks`|`nn.ModuleList[DiTBlock]`，主 Transformer 层。|
|`head`|把 token 输出投影回 patch latent。|
|`freqs`|三维 RoPE 预计算表，分别对应时间、高度、宽度。|
|`img_emb`|可选，CLIP feature 到 DiT hidden size。|
|`ref_conv`|可选，参考图 latent token 化。|
|`control_adapter`|可选，相机控制 latent 注入。|
|`require_vae_embedding` / `require_clip_embedding`|控制 pipeline 是否构造 `y` 和 `clip_feature`。|
|`fuse_vae_embedding_in_latents`|Wan2.2 TI2V 首帧 latent 融合开关。|
|`wantodance_*`|WanToDance 相关扩展开关和子模块。|

### DiT 子模块

|类/函数|签名|用途|输入|输出|
|-|-|-|-|-|
|`flash_attention`|`flash_attention(q, k, v, num_heads, compatibility_mode=False)`|按可用库选择 FlashAttention 3、FlashAttention 2、SageAttention 或 PyTorch SDPA。|`q,k,v: [B,S,D]`。|`[B,S,D]`。|
|`modulate`|`modulate(x, shift, scale)`|AdaLN 调制。|特征、shift、scale。|调制后特征。|
|`sinusoidal_embedding_1d`|`sinusoidal_embedding_1d(dim, position)`|生成一维正弦 timestep embedding。|维度和 position tensor。|`[N,dim]`。|
|`precompute_freqs_cis_3d`|`precompute_freqs_cis_3d(dim, end=1024, theta=10000.0)`|生成 3D RoPE 的时间/高/宽复数表。|head 维度、最大长度。|`(f_freqs,h_freqs,w_freqs)`。|
|`rope_apply`|`rope_apply(x, freqs, num_heads)`|对 Q/K 应用 RoPE。|`x: [B,S,D]`，复数 freqs。|`[B,S,D]`。|
|`RMSNorm.forward`|`forward(x)`|RMS normalization。|任意最后一维为 `dim` 的 tensor。|同形状 tensor。|
|`SelfAttention.forward`|`forward(x, freqs)`|Wan 自注意力。|token、RoPE 表。|token。|
|`CrossAttention.forward`|`forward(x, y)`|文本/图像 cross attention。|latent token 和 context token。|latent token。|
|`DiTBlock.forward`|`forward(x, context, t_mod, freqs)`|self-attn、cross-attn、MLP 和 timestep gate。|latent token、context、调制、RoPE。|latent token。|
|`Head.forward`|`forward(x, t_mod)`|输出 projection。|token 和 timestep embedding。|patch 输出。|

### 其他模型类

|类|主要签名|用途|输入|输出|
|-|-|-|-|-|
|`WanVideoVAE`|`encode(videos, device, tiled=False, tile_size=(34,34), tile_stride=(18,16))`; `decode(hidden_states, device, tiled=False, tile_size=(34,34), tile_stride=(18,16))`; `encode_framewise(videos, device)`; `decode_framewise(hidden_states, device)`|视频像素和 latent 之间的转换。|视频 `[B,3,T,H,W]` 或 latent `[B,C,T_lat,H_lat,W_lat]`。|latent 或视频 tensor。|
|`WanTextEncoder`|`forward(ids, mask=None)`|T5 文本编码。|token ids、attention mask。|文本 hidden states。|
|`HuggingfaceTokenizer`|`__call__(sequence, return_mask=False, **kwargs)`|包装 `AutoTokenizer`，支持清洗、padding、truncation。|字符串或字符串列表。|`input_ids` 或 `(input_ids, attention_mask)`。|
|`WanImageEncoder`|`encode_image(image)`|CLIP 图像编码。|预处理后的 image tensor 列表。|CLIP context。|
|`VaceWanModel`|`forward(x, vace_context, context, t_mod, freqs, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)`|为指定 DiT 层生成 VACE hint。|当前 token、VACE 条件、文本上下文。|hint 列表。|
|`WanMotionControllerModel`|`forward(motion_bucket_id)`|把 motion bucket 映射为 timestep modulation。|bucket tensor。|`[B,6*dim]`。|
|`MotWanModel`|`forward(block, x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id)`|VAP/MoT 分支与主 DiT block 协同更新。|主分支和 VAP 分支 token。|`(x,x_vap)`。|
|`WanAnimateAdapter`|`after_patch_embedding(x, pose_latents, face_pixel_values)`; `after_transformer_block(block_id, x, motion_vec)`|Animate pose/face 条件注入。|patch 后 token、pose latent、人脸像素。|更新后的 token 或 motion vector。|
|`WanS2VAudioEncoder`|`get_audio_feats_per_inference(input_audio, sample_rate, audio_processor, fps, batch_frames, dtype, device)`|提取每个推理片段的音频 embedding。|音频数组、采样率、processor。|音频 embedding 列表/张量。|
|`LongCatVideoTransformer3DModel`|`forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, num_cond_latents=0, ...)`|LongCat-Video DiT 变体。|latent、timestep、文本条件和 mask。|噪声预测。|

## 关键函数签名与用途

### `WanVideoPipeline.from_pretrained`

```python
WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=get_device_type(),
    model_configs=[],
    tokenizer_config=ModelConfig(...),
    audio_processor_config=None,
    redirect_common_files=True,
    use_usp=False,
    vram_limit=None,
) -> WanVideoPipeline
```

用途：下载并加载 `model_configs` 指定的模型，按 `model_name` 放入 pipeline 属性；可重定向常用 Wan 文件到统一 safetensors 仓库，可初始化 unified sequence parallel，可启用显存管理。

输入：

|参数|含义|
|-|-|
|`model_configs`|`ModelConfig` 列表，每个配置指定模型 ID、文件 pattern、加载 dtype/device、converter 等。|
|`tokenizer_config`|tokenizer 路径配置。|
|`audio_processor_config`|Wav2Vec2 processor 路径配置；S2V 需要。|
|`use_usp`|是否初始化 USP 并替换 attention/DiT/VACE forward。|
|`vram_limit`|显存管理上限。|

输出：配置完毕的 `WanVideoPipeline`。

### `WanVideoPipeline.__call__`

```python
pipe(
    prompt="",
    negative_prompt="",
    input_image=None,
    end_image=None,
    input_video=None,
    denoising_strength=1.0,
    input_audio=None,
    audio_embeds=None,
    audio_sample_rate=16000,
    s2v_pose_video=None,
    s2v_pose_latents=None,
    motion_video=None,
    control_video=None,
    reference_image=None,
    camera_control_direction=None,
    camera_control_speed=1/54,
    camera_control_origin=(...),
    vace_video=None,
    vace_video_mask=None,
    vace_reference_image=None,
    vace_scale=1.0,
    animate_pose_video=None,
    animate_face_video=None,
    animate_inpaint_video=None,
    animate_mask_video=None,
    vap_video=None,
    vap_prompt=" ",
    negative_vap_prompt=" ",
    seed=None,
    rand_device="cpu",
    height=480,
    width=832,
    num_frames=81,
    cfg_scale=5.0,
    cfg_merge=False,
    switch_DiT_boundary=0.875,
    num_inference_steps=50,
    sigma_shift=5.0,
    motion_bucket_id=None,
    longcat_video=None,
    tiled=True,
    tile_size=(30, 52),
    tile_stride=(15, 26),
    sliding_window_size=None,
    sliding_window_stride=None,
    tea_cache_l1_thresh=None,
    tea_cache_model_id="",
    wantodance_music_path=None,
    wantodance_reference_image=None,
    wantodance_fps=30,
    wantodance_keyframes=None,
    wantodance_keyframes_mask=None,
    framewise_decoding=False,
    progress_bar_cmd=tqdm,
    output_type="quantized",
) -> list[PIL.Image.Image] | torch.Tensor
```

用途：统一执行 T2V、I2V、FLF2V、V2V、Control、Camera Control、VACE、Animate、VAP、S2V、LongCat、WanToDance 等 Wan 系列推理路径。

输出：

|`output_type`|返回|
|-|-|
|`"quantized"`|`list[PIL.Image.Image]`，适合 `save_video`。|
|`"floatpoint"`|VAE 解码后的浮点 tensor，通常范围约为 `[-1,1]`。|

### `model_fn_wan_video`

```python
model_fn_wan_video(
    dit,
    motion_controller=None,
    vace=None,
    vap=None,
    animate_adapter=None,
    latents=None,
    timestep=None,
    context=None,
    clip_feature=None,
    y=None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds=None,
    motion_latents=None,
    s2v_pose_latents=None,
    vap_hidden_state=None,
    vap_clip_feature=None,
    context_vap=None,
    drop_motion_frames=True,
    tea_cache=None,
    use_unified_sequence_parallel=False,
    motion_bucket_id=None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size=None,
    sliding_window_stride=None,
    cfg_merge=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents=False,
    wantodance_refimage_feature=None,
    wantodance_fps=30.0,
    music_feature=None,
    skip_9th_layer=False,
    memory_size=0,
    **kwargs,
) -> torch.Tensor
```

用途：Wan 去噪模型统一入口。根据输入条件自动分派：

|条件|分支|
|-|-|
|`sliding_window_size` 和 `sliding_window_stride` 非空|使用 `TemporalTiler_BCTHW.run` 分窗口去噪并融合。|
|`dit` 是 `LongCatVideoTransformer3DModel`|调用 `model_fn_longcat_video`。|
|`audio_embeds is not None`|调用 `model_fn_wans2v`。|
|`vap is not None`|在指定 block 上调用 `MotWanModel` 协同更新 VAP 分支。|
|`vace_context is not None`|先计算 VACE hint，再注入映射的 DiT block。|
|`pose_latents` 和 `face_pixel_values` 非空|通过 `WanAnimateAdapter` 注入 Animate 条件。|
|`memory_size > 0`|StoryMem 分支，前缀 memory token 使用负时间 RoPE。|

输出：噪声预测，形状与 `latents` 一致。

### `storymem_select_temporal_freqs`

```python
storymem_select_temporal_freqs(
    freqs_0: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor
```

用途：为 StoryMem 的负时间位置选择 RoPE 频率。正时间位置直接索引 `freqs_0`；负时间位置取绝对值索引后做复共轭，实现 `cis(-t)=conj(cis(t))`。

输入：`freqs_0` 是时间维 RoPE 表，`positions` 是包含负 memory 位置和非负当前视频位置的一维整数 tensor。

输出：与 `positions` 对齐的时间 RoPE 表。

### `TemporalTiler_BCTHW`

```python
class TemporalTiler_BCTHW:
    def build_1d_mask(self, length, left_bound, right_bound, border_width)
    def build_mask(self, data, is_bound, border_width)
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None)
```

用途：对 `[B,C,T,H,W]` 张量按时间窗口切片，分别调用 `model_fn`，再用边界 mask 加权融合重叠区域。

输出：完整时间长度的模型预测。

### `TeaCache`

```python
class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id)
    def check(self, dit: WanModel, x, t_mod) -> bool
    def store(self, hidden_states)
    def update(self, hidden_states) -> torch.Tensor
```

用途：基于 timestep modulation 的相对 L1 距离判断当前 step 是否复用上一轮 residual，以减少 DiT block 计算。

输出：`check` 返回 `True` 表示复用缓存；`update` 返回加上缓存 residual 的 hidden states。

## 外部库函数

下表列出 Wan Video 相关代码中高频且影响数据形状/语义的库函数。签名采用本文所用形式，不展开所有可选参数。

### PyTorch

|库函数|常用签名|用途|输入|输出|
|-|-|-|-|-|
|`torch.no_grad`|`@torch.no_grad()`|禁用梯度，降低推理显存。|被装饰函数。|无梯度执行的函数。|
|`torch.Tensor.to`|`tensor.to(device=None, dtype=None)`|移动设备或转换 dtype。|tensor、device、dtype。|转换后的 tensor。|
|`torch.cat` / `torch.concat`|`torch.cat(tensors, dim=0)`|按维度拼接 tensor。|tensor 序列。|拼接 tensor。|
|`torch.stack`|`torch.stack(tensors, dim=0)`|新增维度堆叠。|同形状 tensor 序列。|高一维 tensor。|
|`torch.zeros`|`torch.zeros(size, dtype=None, device=None)`|创建全零 tensor。|形状、dtype、device。|tensor。|
|`torch.ones`|`torch.ones(size, dtype=None, device=None)`|创建全一 tensor。|形状、dtype、device。|tensor。|
|`torch.randn`|`torch.randn(size, dtype=None, device=None)`|创建高斯噪声。|形状、dtype、device。|tensor。|
|`torch.arange`|`torch.arange(start=0, end, step=1, device=None)`|创建连续索引，StoryMem RoPE 位置使用。|范围。|一维 tensor。|
|`torch.outer`|`torch.outer(input, vec2)`|构建正弦 embedding 或 RoPE 角度矩阵。|两个一维 tensor。|二维 tensor。|
|`torch.polar`|`torch.polar(abs, angle)`|根据幅值和相位构造复数 RoPE。|幅值、角度。|复数 tensor。|
|`torch.view_as_complex`|`torch.view_as_complex(input)`|把最后一维为 2 的实数 tensor 视为复数。|实数 tensor。|复数 tensor。|
|`torch.view_as_real`|`torch.view_as_real(input)`|把复数 tensor 转回实部/虚部。|复数 tensor。|最后一维为 2 的实数 tensor。|
|`torch.chunk`|`torch.chunk(input, chunks, dim=0)`|按维度切块；CFG、USP 和 S2V 多片段使用。|tensor、块数。|tuple[tensor]。|
|`torch.repeat_interleave`|`torch.repeat_interleave(input, repeats, dim=None)`|重复帧或 mask，匹配 Wan VAE 时间打包。|tensor、重复次数。|tensor。|
|`torch.nn.functional.scaled_dot_product_attention`|`F.scaled_dot_product_attention(query, key, value, ...)`|PyTorch attention fallback。|`[B,H,S,D]` Q/K/V。|attention 输出。|
|`torch.nn.functional.interpolate`|`F.interpolate(input, size=None, scale_factor=None, mode=...)`|调整 mask、音频特征或 VACE mask 尺寸。|NCHW/NCTHW tensor。|重采样 tensor。|
|`torch.utils.checkpoint.checkpoint`|`checkpoint(function, *args, use_reentrant=False)`|梯度检查点，训练时节省显存。|函数和参数。|函数输出。|
|`torch.autograd.graph.save_on_cpu`|`with save_on_cpu(): ...`|梯度检查点 offload 到 CPU。|上下文。|上下文内保存策略改变。|
|`torch.distributed.is_initialized`|`dist.is_initialized()`|判断分布式是否可用。|无。|bool。|
|`torch.distributed.get_world_size`|`dist.get_world_size()`|获取并行进程数。|无。|int。|

### `torch.nn`

|库函数/类|常用签名|用途|输入|输出|
|-|-|-|-|-|
|`nn.Module`|`class MyModule(nn.Module)`|模型基类。|子模块定义。|可调用模型对象。|
|`nn.Linear`|`nn.Linear(in_features, out_features, bias=True)`|线性投影。|`[...,in_features]`。|`[...,out_features]`。|
|`nn.Conv3d`|`nn.Conv3d(in_channels, out_channels, kernel_size, stride=...)`|3D patch embedding 或 VAE 卷积。|`[B,C,T,H,W]`。|卷积输出。|
|`nn.Conv2d`|`nn.Conv2d(in_channels, out_channels, kernel_size, stride=...)`|参考图 latent patch 化。|`[B,C,H,W]`。|卷积输出。|
|`nn.LayerNorm`|`nn.LayerNorm(normalized_shape, eps=...)`|最后维归一化。|tensor。|同形状 tensor。|
|`nn.GELU`|`nn.GELU(approximate='tanh')`|MLP 激活。|tensor。|tensor。|
|`nn.SiLU`|`nn.SiLU()`|timestep MLP 激活。|tensor。|tensor。|
|`nn.ModuleList`|`nn.ModuleList(modules)`|保存 block 列表。|模块列表。|可注册子模块容器。|
|`nn.Sequential`|`nn.Sequential(*modules)`|顺序组合网络。|模块列表。|可调用容器。|
|`nn.Parameter`|`nn.Parameter(data)`|注册可训练参数。|tensor。|Parameter。|

### 图像、数组、形状变换

|库函数|常用签名|用途|输入|输出|
|-|-|-|-|-|
|`PIL.Image.Image.resize`|`image.resize((width, height))`|把输入图像调整到推理尺寸。|PIL image。|PIL image。|
|`numpy.poly1d`|`np.poly1d(coefficients)`|TeaCache 阈值重标定多项式。|系数列表。|可调用多项式对象。|
|`numpy.concatenate`|`np.concatenate(arrays, axis=...)`|拼接 WanToDance 音乐特征。|ndarray 序列。|ndarray。|
|`numpy.zeros_like`|`np.zeros_like(a, dtype=...)`|创建 onset/beat one-hot。|ndarray。|ndarray。|
|`einops.rearrange`|`rearrange(tensor, pattern, **axes_lengths)`|显式变换张量维度。|tensor 和 pattern。|重排 tensor。|
|`einops.repeat`|`repeat(tensor, pattern, **axes_lengths)`|扩展 mask 维度。|tensor 和 pattern。|重复后的 tensor。|
|`tqdm.tqdm`|`tqdm(iterable)`|显示扩散进度。|可迭代对象。|包装后的迭代器。|

### Transformers、音频和注意力加速

|库函数|常用签名|用途|输入|输出|
|-|-|-|-|-|
|`transformers.AutoTokenizer.from_pretrained`|`AutoTokenizer.from_pretrained(name, **kwargs)`|加载 Wan 文本 tokenizer。|本地路径或模型名。|tokenizer。|
|`transformers.Wav2Vec2Processor.from_pretrained`|`Wav2Vec2Processor.from_pretrained(path)`|加载 S2V 音频 processor。|路径。|processor。|
|`flash_attn_interface.flash_attn_func`|`flash_attn_func(q, k, v, ...)`|FlashAttention 3。|`[B,S,H,D]` Q/K/V。|attention 输出。|
|`flash_attn.flash_attn_func`|`flash_attn_func(q, k, v, ...)`|FlashAttention 2。|`[B,S,H,D]` Q/K/V。|attention 输出。|
|`sageattention.sageattn`|`sageattn(q, k, v, ...)`|SageAttention fallback。|`[B,H,S,D]` Q/K/V。|attention 输出。|
|`librosa.load`|`librosa.load(path, sr=...)`|读取并重采样音乐。|音频路径、采样率。|`(waveform, sample_rate)`。|
|`librosa.onset.onset_strength`|`onset_strength(y, sr, ...)`|计算 onset envelope。|波形、采样率。|一维特征。|
|`librosa.feature.mfcc`|`mfcc(y, sr, n_mfcc=20)`|计算 MFCC。|波形、采样率。|`[n_mfcc, frames]`。|
|`librosa.feature.chroma_cens`|`chroma_cens(y, sr, hop_length, n_chroma=12)`|计算 chroma 特征。|波形、采样率。|`[n_chroma, frames]`。|
|`librosa.onset.onset_detect`|`onset_detect(onset_envelope, sr, hop_length)`|检测 onset 峰值。|onset envelope。|峰值索引。|
|`librosa.beat.tempo`|`tempo(y)`|估计 BPM。|波形。|BPM 数组。|
|`librosa.beat.beat_track`|`beat_track(onset_envelope, sr, hop_length, start_bpm, tightness)`|检测节拍位置。|onset envelope、BPM。|`(tempo, beat_indices)`。|

### DiffSynth 内部 API

|函数/类|签名摘要|用途|输入|输出|
|-|-|-|-|-|
|`BasePipeline.download_and_load_models`|`download_and_load_models(model_configs, vram_limit)`|下载并加载模型池。|`ModelConfig` 列表。|model pool。|
|`BasePipeline.load_models_to_device`|`load_models_to_device(model_names)`|按显存管理策略 onload/offload 模型。|模型名列表。|无。|
|`BasePipeline.unit_runner`|`unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)`|执行 PipelineUnit 并合并输出。|unit 和输入字典。|更新后的三个字典。|
|`BasePipeline.preprocess_image`|`preprocess_image(image, min_value=-1, max_value=1)`|PIL 图像转模型输入 tensor。|PIL image。|`[1,3,H,W]`。|
|`BasePipeline.preprocess_video`|`preprocess_video(video, min_value=-1, max_value=1)`|图像帧列表或 tensor 转视频 tensor。|帧列表/tensor。|`[B,3,T,H,W]`。|
|`BasePipeline.generate_noise`|`generate_noise(shape, seed=None, rand_device='cpu')`|生成可复现噪声。|形状、seed、设备。|noise tensor。|
|`BasePipeline.vae_output_to_video`|`vae_output_to_video(video)`|把 VAE tensor 转 PIL 帧。|`[B,3,T,H,W]`。|帧列表。|
|`FlowMatchScheduler.set_timesteps`|`set_timesteps(num_inference_steps, denoising_strength=1.0, shift=...)`|生成推理时间步。|步数、强度、shift。|写入 `scheduler.timesteps`。|
|`FlowMatchScheduler.add_noise`|`add_noise(original_samples, noise, timestep)`|把干净 latent 加噪到指定时间。|latent、noise、timestep。|noisy latent。|
|`FlowMatchScheduler.step`|`step(model_output, timestep, sample)`|执行一次 flow matching 更新。|噪声预测、时间步、当前 latent。|下一步 latent。|
|`gradient_checkpoint_forward`|`gradient_checkpoint_forward(module, use_gradient_checkpointing, use_gradient_checkpointing_offload, *args)`|统一训练/推理 forward 与 checkpoint 逻辑。|模块、开关、参数。|模块输出。|

## 扩展注意事项

1. 新增条件时优先写成 `PipelineUnit`，让输入构造保持可组合；只有需要同时改写正/负/共享字典时才使用 `take_over=True`。
2. 新增模型分支时优先在 `model_fn_wan_video` 里用清晰条件分派，保证普通 Wan 路径不受影响。
3. 修改时间维时要同步处理 `noise`、`latents`、`input_latents`、`y`、RoPE `freqs` 和 VAE mask。StoryMem 分支就是这个模式。
4. 任何新增 `y` 条件都要确认 `dit.in_dim`、`latents.shape[1]` 和条件通道数匹配。
5. `tiled=True` 只影响 VAE 编解码，不改变 DiT 的空间分辨率；滑动窗口则影响 DiT 时间维推理。
6. `cfg_merge=True` 会把正负条件合并到 batch 维，新增条件若参与 CFG，需要同时支持正负字典或共享字典复制。
