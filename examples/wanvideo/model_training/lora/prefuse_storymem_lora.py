import argparse
import glob
import os

import torch
from safetensors.torch import load_file, save_file

from diffsynth.utils.lora import GeneralLoRALoader


PREFIXES_TO_STRIP = (
    "base_model.model.",
    "module.base_model.model.",
)


def strip_storymem_prefixes(state_dict):
    converted = {}
    for key, value in state_dict.items():
        for prefix in PREFIXES_TO_STRIP:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        converted[key] = value
    return converted


def expand_paths(paths):
    expanded = []
    for path in paths:
        matches = sorted(glob.glob(path))
        expanded.extend(matches or [path])
    return expanded


def prepare_device(device):
    if device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise ImportError("Using --compute_device npu requires torch-npu to be installed.") from exc
    return torch.device(device)


def dtype_from_name(name):
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }[name]


def load_lora(lora_path):
    state_dict = load_file(lora_path, device="cpu")
    state_dict = strip_storymem_prefixes(state_dict)
    return GeneralLoRALoader().convert_state_dict(state_dict)


def fuse_weight(base_weight, lora_up, lora_down, alpha, compute_device, compute_dtype):
    base_dtype = base_weight.dtype
    base_device = base_weight.device
    base_weight = base_weight.to(device=compute_device, dtype=compute_dtype)
    lora_up = lora_up.to(device=compute_device, dtype=compute_dtype)
    lora_down = lora_down.to(device=compute_device, dtype=compute_dtype)
    if len(lora_up.shape) == 4:
        lora_up = lora_up.squeeze(3).squeeze(2)
        lora_down = lora_down.squeeze(3).squeeze(2)
        delta = torch.mm(lora_up, lora_down).unsqueeze(2).unsqueeze(3)
    else:
        delta = torch.mm(lora_up, lora_down)
    return (base_weight + alpha * delta).to(device=base_device, dtype=base_dtype)


def release_accelerator_cache(device):
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
        torch.npu.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def prefuse_lora(base_shards, lora_path, output_dir, alpha, compute_device, compute_dtype):
    base_shards = expand_paths(base_shards)
    if not base_shards:
        raise ValueError("No base shards were provided.")
    for path in base_shards:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Base shard not found: {path}")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    os.makedirs(output_dir, exist_ok=True)
    lora = load_lora(lora_path)
    lora_layer_names = {
        key.replace(".lora_B.weight", "")
        for key in lora
        if key.endswith(".lora_B.weight")
    }

    total_fused = 0
    for shard_path in base_shards:
        print(f"Loading base shard: {shard_path}", flush=True)
        shard = load_file(shard_path, device="cpu")
        fused_in_shard = 0
        for name in sorted(lora_layer_names):
            weight_key = f"{name}.weight"
            if weight_key not in shard:
                continue
            shard[weight_key] = fuse_weight(
                shard[weight_key],
                lora[f"{name}.lora_B.weight"],
                lora[f"{name}.lora_A.weight"],
                alpha,
                compute_device,
                compute_dtype,
            )
            release_accelerator_cache(compute_device)
            fused_in_shard += 1

        output_path = os.path.join(output_dir, os.path.basename(shard_path))
        save_file(shard, output_path)
        release_accelerator_cache(compute_device)
        total_fused += fused_in_shard
        print(f"Saved {output_path}; fused {fused_in_shard} tensors.", flush=True)

    unmatched = len(lora_layer_names) - total_fused
    print(
        f"Done. Fused {total_fused}/{len(lora_layer_names)} LoRA tensors "
        f"into {len(base_shards)} shard(s). Unmatched LoRA tensors: {unmatched}.",
        flush=True,
    )
    if total_fused == 0:
        raise RuntimeError("No tensors were fused. Check that the LoRA keys match the base model keys.")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline-fuse a StoryMem preset LoRA into Wan DiT safetensors shards.")
    parser.add_argument("--base_shards", nargs="+", required=True, help="Base DiT safetensors shard paths or glob patterns.")
    parser.add_argument("--lora_path", required=True, help="StoryMem preset LoRA safetensors path.")
    parser.add_argument("--output_dir", required=True, help="Directory for fused DiT safetensors shards.")
    parser.add_argument("--alpha", type=float, default=1.0, help="LoRA scale used for fusion.")
    parser.add_argument("--compute_device", default="cpu", help="Device used for LoRA matrix multiplications, e.g. cpu or npu:0.")
    parser.add_argument("--compute_dtype", default="bf16", choices=("bf16", "bfloat16", "fp32", "float32", "fp16", "float16"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prefuse_lora(
        base_shards=args.base_shards,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        alpha=args.alpha,
        compute_device=prepare_device(args.compute_device),
        compute_dtype=dtype_from_name(args.compute_dtype),
    )
