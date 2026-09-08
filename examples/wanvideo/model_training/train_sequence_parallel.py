"""Experimental Wan training entry point using one world-sized USP group.

This file intentionally lives beside, rather than modifies, train.py.  Every
rank consumes the same sample and resets its RNG to the same per-step seed;
only the DiT token sequence is sharded by xFuser USP.
"""

import random
import time

import accelerate
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

import train as standard_train
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import (
    ImageCropAndResize,
    LoadAudio,
    LoadImage,
    LoadVideo,
    ToAbsolutePath,
)
from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.runner import (
    debug_memory_snapshot,
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
    save_training_args,
)


def initialize_usp_from_accelerate(accelerator, sequence_parallel_size):
    """Initialize xFuser groups without reinitializing torch.distributed."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Sequence-parallel training requires a distributed launch.")

    world_size = dist.get_world_size()
    if sequence_parallel_size != world_size:
        raise ValueError(
            "This isolated training entry currently supports one USP group only: "
            f"sequence_parallel_size ({sequence_parallel_size}) must equal world_size ({world_size})."
        )

    try:
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
    except ImportError as exc:
        raise RuntimeError(
            'USP dependencies are missing. Install "xfuser[flash-attn]>=0.4.3".'
        ) from exc

    init_distributed_environment(rank=dist.get_rank(), world_size=world_size)
    initialize_model_parallel(
        sequence_parallel_degree=sequence_parallel_size,
        ring_degree=1,
        ulysses_degree=sequence_parallel_size,
    )
    accelerator.wait_for_everyone()


class SequenceParallelWanTrainingModule(standard_train.WanTrainingModule):
    def __init__(self, *args, sequence_parallel_seed=0, **kwargs):
        self.sequence_parallel_seed = sequence_parallel_seed
        self.sequence_parallel_step = 0
        super().__init__(*args, **kwargs)
        self.pipe.enable_usp()

        sp_size = dist.get_world_size()
        for name in ("dit", "dit2"):
            dit = getattr(self.pipe, name, None)
            if dit is not None and dit.blocks[0].self_attn.num_heads % sp_size != 0:
                raise ValueError(
                    f"{name} num_heads ({dit.blocks[0].self_attn.num_heads}) must be "
                    f"divisible by sequence_parallel_size ({sp_size})."
                )

    def _reset_synchronized_rng(self):
        seed = self.sequence_parallel_seed + self.sequence_parallel_step
        self.sequence_parallel_step += 1
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        device = torch.device(self.pipe.device)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        elif device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.manual_seed(seed)

    def forward(self, data, inputs=None):
        self._reset_synchronized_rng()
        return super().forward(data, inputs=inputs)


def launch_sequence_parallel_training(accelerator, dataset, model, model_logger, args):
    if args.enable_model_cpu_offload:
        raise ValueError(
            "The isolated USP entry does not yet support --enable_model_cpu_offload."
        )

    if accelerator.is_main_process:
        save_training_args(args)

    optimizer_class = get_optimizer_class(args.customized_optimizer)
    optimizer = optimizer_class(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # Every SP rank must consume the same sample in the same order.  The
    # prepared loader is supplied only so Accelerate/DeepSpeed can infer its
    # batch configuration; training deliberately iterates the raw loader.
    generator = torch.Generator().manual_seed(args.sequence_parallel_seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda batch: batch[0],
        num_workers=0,
        generator=generator,
    )

    model.to(device=accelerator.device)
    model, optimizer, _prepared_loader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)

    for epoch_id in range(args.num_epochs):
        for data in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                try:
                    debug_memory_snapshot(model, "before_backward")
                    accelerator.backward(loss)
                    debug_memory_snapshot(model, "after_backward")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model_logger.on_step_end(
                        accelerator, model, args.save_steps, loss=loss
                    )
                except Exception:
                    debug_memory_snapshot(model, "on_exception")
                    raise
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, args.save_steps)


def main():
    script_start = time.time()
    parser = standard_train.wan_parser()
    parser.add_argument(
        "--sequence_parallel_size",
        type=int,
        required=True,
        help="USP degree. This entry currently requires it to equal world size.",
    )
    parser.add_argument(
        "--sequence_parallel_seed",
        type=int,
        default=0,
        help="Base seed shared by all ranks; incremented once per training forward.",
    )
    args = parser.parse_args()
    if args.temporal_rope_target_num_frames is not None:
        if args.temporal_rope_target_num_frames < 1 or args.temporal_rope_target_num_frames % 4 != 1:
            parser.error("--temporal_rope_target_num_frames must be a positive 4n+1 frame count.")
        if args.temporal_rope_target_num_frames < args.num_frames:
            parser.error("--temporal_rope_target_num_frames must be at least --num_frames.")

    standard_train._enable_checkpoint_tracebacks(
        args.debug_checkpoints, args.debug_checkpoint_trace_after
    )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    initialize_usp_from_accelerate(accelerator, args.sequence_parallel_size)

    # Keep model initialization and LoRA initialization identical on every SP rank.
    random.seed(args.sequence_parallel_seed)
    np.random.seed(args.sequence_parallel_seed % (2**32))
    torch.manual_seed(args.sequence_parallel_seed)

    data_file_keys = [key for key in args.data_file_keys.split(",") if key]
    extra_inputs = [] if args.extra_inputs is None else [
        key for key in args.extra_inputs.split(",") if key
    ]
    if "memory_images" in extra_inputs and "memory_images" not in data_file_keys:
        raise ValueError(
            "Using --extra_inputs memory_images requires --data_file_keys to include memory_images."
        )
    if "input_image" in extra_inputs and "input_image" not in data_file_keys:
        raise ValueError(
            "Using --extra_inputs input_image requires --data_file_keys to include input_image."
        )

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
            "animate_face_video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(
                args.num_frames,
                4,
                1,
                frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "input_image": ToAbsolutePath(args.dataset_base_path)
            >> LoadImage()
            >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
            "memory_images": standard_train.LoadStoryMemMemoryImages(
                args.dataset_base_path, args.height, args.width, args.max_pixels
            ),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        },
    )

    model = SequenceParallelWanTrainingModule(
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
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        training_scheduler_shift=args.training_scheduler_shift,
        temporal_rope_target_num_frames=args.temporal_rope_target_num_frames,
        debug_memory=args.debug_memory,
        debug_memory_units=args.debug_memory_units,
        debug_memory_tensors_topk=args.debug_memory_tensors_topk,
        debug_memory_summary=args.debug_memory_summary,
        debug_checkpoints=args.debug_checkpoints,
        sequence_parallel_seed=args.sequence_parallel_seed,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launch_sequence_parallel_training(accelerator, dataset, model, model_logger, args)
    standard_train._debug_checkpoint(
        args.debug_checkpoints,
        "sequence_parallel_training:done",
        script_start,
        accelerator.device,
    )


if __name__ == "__main__":
    main()
