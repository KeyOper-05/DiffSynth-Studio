import torch, os, argparse, accelerate, warnings, json, time, sys, faulthandler
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, LoadImage, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _current_rss_mib():
    status_path = "/proc/self/status"
    if not os.path.exists(status_path):
        return "unknown"
    try:
        with open(status_path) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    value_kib = int(line.split()[1])
                    return f"{value_kib / 1024:.2f} MiB"
    except Exception:
        return "unknown"
    return "unknown"


def _debug_checkpoint(enabled, tag, start_time=None, device=None):
    if not enabled:
        return
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    elapsed = ""
    if start_time is not None:
        elapsed = f" elapsed={time.time() - start_time:.2f}s"
    device_text = f" device={device}" if device is not None else ""
    print(
        f"[CHECKPOINT][rank {rank} local {local_rank} pid {os.getpid()}]"
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][{tag}]"
        f"{elapsed}{device_text} rss={_current_rss_mib()}",
        flush=True,
    )


def _enable_checkpoint_tracebacks(enabled, seconds):
    if not enabled or seconds <= 0:
        return
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(seconds, repeat=True, file=sys.stderr)
    _debug_checkpoint(True, f"faulthandler_enabled:{seconds}s")


def _format_mib(num_bytes):
    return f"{num_bytes / (1024 ** 2):.2f} MiB"


def _memory_backend(device):
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError):
        return None, None
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda, device
    if device.type == "npu" and hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return torch.npu, device
        except (AttributeError, RuntimeError):
            return None, None
    return None, None


def _call_memory_api(backend, name, device=None, default=0):
    fn = getattr(backend, name, None)
    if fn is None:
        return default
    try:
        return fn(device) if device is not None else fn()
    except TypeError:
        try:
            return fn()
        except Exception:
            return default
    except Exception:
        return default


def _mem_get_info(backend, device):
    fn = getattr(backend, "mem_get_info", None)
    if fn is None:
        return None, None
    try:
        return fn(device)
    except TypeError:
        try:
            return fn()
        except Exception:
            return None, None
    except Exception:
        return None, None


def _iter_named_tensors(value, prefix):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_named_tensors(item, next_prefix)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_named_tensors(item, f"{prefix}[{index}]")


class LoadStoryMemMemoryImages:
    def __init__(self, base_path, height=None, width=None, max_pixels=1920 * 1080):
        self.image_operator = (
            ToAbsolutePath(base_path)
            >> LoadImage()
            >> ImageCropAndResize(height, width, max_pixels, 16, 16)
        )

    def _parse_paths(self, data):
        if isinstance(data, str):
            paths = json.loads(data)
        elif isinstance(data, list):
            paths = data
        else:
            raise ValueError("metadata field 'memory_images' must be a non-empty JSON list string or list of image paths.")
        if not isinstance(paths, list) or len(paths) == 0:
            raise ValueError("metadata field 'memory_images' must contain at least one image path.")
        if not all(isinstance(path, str) and path.strip() for path in paths):
            raise ValueError("metadata field 'memory_images' must contain only non-empty string paths.")
        return paths

    def __call__(self, data):
        return [self.image_operator(path.strip()) for path in self._parse_paths(data)]



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        training_scheduler_shift=None,
        temporal_rope_target_num_frames=None,
        debug_memory=False,
        debug_memory_units=False,
        debug_memory_tensors_topk=20,
        debug_memory_summary=False,
        debug_checkpoints=False,
    ):
        super().__init__()
        init_start = time.time()
        _debug_checkpoint(debug_checkpoints, "WanTrainingModule:init:start", init_start, device)
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Load models
        step_start = time.time()
        _debug_checkpoint(debug_checkpoints, "parse_model_configs:start", init_start, device)
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        _debug_checkpoint(debug_checkpoints, "parse_model_configs:done", step_start, device)
        step_start = time.time()
        _debug_checkpoint(debug_checkpoints, "tokenizer_config:start", init_start, device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        _debug_checkpoint(debug_checkpoints, "tokenizer_config:done", step_start, device)
        step_start = time.time()
        _debug_checkpoint(debug_checkpoints, "WanVideoPipeline.from_pretrained:start", init_start, device)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        _debug_checkpoint(debug_checkpoints, "WanVideoPipeline.from_pretrained:done", step_start, device)
        step_start = time.time()
        _debug_checkpoint(debug_checkpoints, "split_pipeline_units:start", init_start, device)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        _debug_checkpoint(debug_checkpoints, "split_pipeline_units:done", step_start, device)
        step_start = time.time()
        _debug_checkpoint(debug_checkpoints, "resume_from_checkpoint:start", init_start, device)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        _debug_checkpoint(debug_checkpoints, "resume_from_checkpoint:done", step_start, device)
        
        # Training mode
        step_start = time.time()
        _debug_checkpoint(debug_checkpoints, "switch_pipe_to_training_mode:start", init_start, device)
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            training_scheduler_shift=training_scheduler_shift,
            task=task,
        )
        _debug_checkpoint(debug_checkpoints, "switch_pipe_to_training_mode:done", step_start, device)
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.debug_memory = debug_memory
        self.debug_memory_units = debug_memory_units
        self.debug_memory_tensors_topk = debug_memory_tensors_topk
        self.debug_memory_summary = debug_memory_summary
        self.debug_checkpoints = debug_checkpoints
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.training_scheduler_shift = training_scheduler_shift
        self.temporal_rope_target_num_frames = temporal_rope_target_num_frames
        _debug_checkpoint(debug_checkpoints, "WanTrainingModule:init:done", init_start, device)

    def _debug_memory_snapshot(self, tag, inputs=None):
        backend, device = _memory_backend(self.pipe.device)
        if not self.debug_memory or backend is None:
            return
        _call_memory_api(backend, "synchronize", device, default=None)
        allocated = _call_memory_api(backend, "memory_allocated", device)
        reserved = _call_memory_api(backend, "memory_reserved", device)
        peak_allocated = _call_memory_api(backend, "max_memory_allocated", device)
        peak_reserved = _call_memory_api(backend, "max_memory_reserved", device)
        free, total = _mem_get_info(backend, device)
        rank = os.environ.get("LOCAL_RANK", "0")
        allocated_ratio = f"{allocated / total * 100:.2f}% total" if total else "total unknown"
        reserved_ratio = f"{reserved / total * 100:.2f}% total" if total else "total unknown"
        free_text = _format_mib(free) if free is not None else "unknown"
        total_text = _format_mib(total) if total is not None else "unknown"
        print(
            f"[VRAM][{device.type}][rank {rank}][{tag}] "
            f"allocated={_format_mib(allocated)} ({allocated_ratio}), "
            f"reserved={_format_mib(reserved)} ({reserved_ratio}), "
            f"peak_allocated={_format_mib(peak_allocated)}, "
            f"peak_reserved={_format_mib(peak_reserved)}, "
            f"free={free_text}, total={total_text}",
            flush=True,
        )
        if inputs is not None and self.debug_memory_tensors_topk != 0:
            seen = set()
            tensors = []
            for scope, value in zip(("shared", "posi", "nega"), inputs):
                for name, tensor in _iter_named_tensors(value, scope):
                    if id(tensor) in seen:
                        continue
                    seen.add(id(tensor))
                    nbytes = tensor.numel() * tensor.element_size()
                    tensors.append((nbytes, name, tuple(tensor.shape), str(tensor.dtype), str(tensor.device)))
            tensors.sort(reverse=True, key=lambda item: item[0])
            topk = len(tensors) if self.debug_memory_tensors_topk < 0 else min(self.debug_memory_tensors_topk, len(tensors))
            total_input_bytes = sum(item[0] for item in tensors)
            print(f"[VRAM][{device.type}][rank {rank}][{tag}] input_tensors_total={_format_mib(total_input_bytes)} unique_tensors={len(tensors)}", flush=True)
            for nbytes, name, shape, dtype, tensor_device in tensors[:topk]:
                print(f"  [tensor] {name}: shape={shape}, dtype={dtype}, device={tensor_device}, size={_format_mib(nbytes)}", flush=True)
        if self.debug_memory_summary:
            summary_fn = getattr(backend, "memory_summary", None)
            if summary_fn is not None:
                try:
                    print(summary_fn(device=device, abbreviated=True), flush=True)
                except TypeError:
                    try:
                        print(summary_fn(device, abbreviated=True), flush=True)
                    except Exception:
                        print(f"[VRAM][{device.type}][rank {rank}][{tag}] memory_summary unavailable.", flush=True)
                except Exception:
                    print(f"[VRAM][{device.type}][rank {rank}][{tag}] memory_summary unavailable.", flush=True)
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data.get("input_image", data["video"][0])
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        if self.temporal_rope_target_num_frames is not None:
            source_num_frames = len(data["video"])
            if source_num_frames % 4 != 1:
                raise ValueError(f"Sparse temporal RoPE requires 4n+1 training frames, got {source_num_frames}.")
            if self.temporal_rope_target_num_frames < source_num_frames:
                raise ValueError(
                    "temporal_rope_target_num_frames must be at least the sampled video length "
                    f"({self.temporal_rope_target_num_frames} < {source_num_frames})."
                )
            inputs_shared["temporal_rope_target_num_frames"] = self.temporal_rope_target_num_frames
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        backend, device = _memory_backend(self.pipe.device)
        if self.debug_memory and backend is not None:
            _call_memory_api(backend, "reset_peak_memory_stats", device, default=None)
            self._debug_memory_snapshot("before_transfer", inputs)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        self._debug_memory_snapshot("after_transfer", inputs)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            if self.debug_memory_units:
                self._debug_memory_snapshot(f"after_unit:{unit.__class__.__name__}", inputs)
        self._debug_memory_snapshot("before_loss", inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        self._debug_memory_snapshot("after_loss", inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--seed", type=int, default=0, help="Training random seed.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--training_scheduler_shift", type=float, default=None, help="Optional Wan flow-match scheduler shift used to build the training timestep table.")
    parser.add_argument("--temporal_rope_target_num_frames", type=int, default=None, help="Spread sampled training frames over this 4n+1-frame temporal RoPE timeline.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument("--debug_memory", default=False, action="store_true", help="Print CUDA/NPU memory allocator stats during each training forward.")
    parser.add_argument("--debug_memory_units", default=False, action="store_true", help="With --debug_memory, print stats after every WanVideoPipeline unit.")
    parser.add_argument("--debug_memory_tensors_topk", type=int, default=20, help="With --debug_memory, print the largest N live input tensors. Use -1 for all, 0 to disable tensor list.")
    parser.add_argument("--debug_memory_summary", default=False, action="store_true", help="With --debug_memory, also print backend memory_summary when available.")
    parser.add_argument("--debug_checkpoints", default=False, action="store_true", help="Print rank-aware checkpoints around slow initialization and launch stages.")
    parser.add_argument("--debug_checkpoint_trace_after", type=int, default=0, help="With --debug_checkpoints, dump Python stack traces every N seconds while the process is still running.")
    return parser


if __name__ == "__main__":
    script_start = time.time()
    parser = wan_parser()
    args = parser.parse_args()
    if args.temporal_rope_target_num_frames is not None:
        if args.temporal_rope_target_num_frames < 1 or args.temporal_rope_target_num_frames % 4 != 1:
            parser.error("--temporal_rope_target_num_frames must be a positive 4n+1 frame count.")
        if args.temporal_rope_target_num_frames < args.num_frames:
            parser.error("--temporal_rope_target_num_frames must be at least --num_frames.")
    _enable_checkpoint_tracebacks(args.debug_checkpoints, args.debug_checkpoint_trace_after)
    _debug_checkpoint(args.debug_checkpoints, "main:args_parsed", script_start)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    accelerate.utils.set_seed(args.seed, device_specific=True)
    _debug_checkpoint(args.debug_checkpoints, "accelerator:init:done", script_start, accelerator.device)
    data_file_keys = [key for key in args.data_file_keys.split(",") if key]
    extra_inputs = [] if args.extra_inputs is None else [key for key in args.extra_inputs.split(",") if key]
    if "memory_images" in extra_inputs and "memory_images" not in data_file_keys:
        raise ValueError("Using --extra_inputs memory_images requires --data_file_keys to include memory_images.")
    if "input_image" in extra_inputs and "input_image" not in data_file_keys:
        raise ValueError("Using --extra_inputs input_image requires --data_file_keys to include input_image.")
    _debug_checkpoint(args.debug_checkpoints, "dataset:create:start", script_start, accelerator.device)
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=data_file_keys,
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "input_image": ToAbsolutePath(args.dataset_base_path) >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
            "memory_images": LoadStoryMemMemoryImages(args.dataset_base_path, args.height, args.width, args.max_pixels),  # memory_images metadata -> list[PIL.Image] for StoryMem pipeline unit.
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        }
    )
    _debug_checkpoint(args.debug_checkpoints, "dataset:create:done", script_start, accelerator.device)
    _debug_checkpoint(args.debug_checkpoints, "model:create:start", script_start, accelerator.device)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        training_scheduler_shift=args.training_scheduler_shift,
        temporal_rope_target_num_frames=args.temporal_rope_target_num_frames,
        debug_memory=args.debug_memory,
        debug_memory_units=args.debug_memory_units,
        debug_memory_tensors_topk=args.debug_memory_tensors_topk,
        debug_memory_summary=args.debug_memory_summary,
        debug_checkpoints=args.debug_checkpoints,
    )
    _debug_checkpoint(args.debug_checkpoints, "model:create:done", script_start, accelerator.device)
    _debug_checkpoint(args.debug_checkpoints, "model_logger:create:start", script_start, accelerator.device)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    _debug_checkpoint(args.debug_checkpoints, "model_logger:create:done", script_start, accelerator.device)
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    _debug_checkpoint(args.debug_checkpoints, f"launcher:start:{args.task}", script_start, accelerator.device)
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
    _debug_checkpoint(args.debug_checkpoints, f"launcher:done:{args.task}", script_start, accelerator.device)
