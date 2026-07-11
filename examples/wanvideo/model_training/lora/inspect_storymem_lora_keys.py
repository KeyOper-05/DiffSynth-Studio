import argparse
from collections import Counter

try:
    from safetensors import safe_open
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires safetensors. Please run it in the same environment "
        "you use for DiffSynth training, or install safetensors there."
    ) from exc


STORYMEM_MODEL_CONFIG = {
    "model_hash": "5b013604280dd715f8457c6ed6d6a626",
    "model_name": "wan_video_dit",
    "model_class": "diffsynth.models.wan_video_dit.WanModel",
    "extra_kwargs": {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 36,
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "eps": 1e-6,
        "require_clip_embedding": False,
    },
}

LORA_MARKERS = (
    "lora_A",
    "lora_B",
    "lora_down",
    "lora_up",
)

WRAPPER_PREFIXES = (
    "base_model.model.",
    "module.base_model.model.",
    "diffusion_model.",
)


def tensor_shape(reader, key):
    try:
        return tuple(reader.get_slice(key).get_shape())
    except Exception:
        return None


def strip_wrapper_prefix(module_name):
    for prefix in WRAPPER_PREFIXES:
        if module_name.startswith(prefix):
            return module_name[len(prefix):], prefix
    return module_name, None


def parse_lora_module_name(key, strip_prefix=False):
    parts = key.split(".")
    marker_index = None
    for index, part in enumerate(parts):
        if part in LORA_MARKERS:
            marker_index = index
            break
    if marker_index is None:
        return None, None

    module_name = ".".join(parts[:marker_index])
    stripped_prefix = None
    if strip_prefix:
        module_name, stripped_prefix = strip_wrapper_prefix(module_name)
    return module_name, stripped_prefix


def storymem_supported_modules():
    num_layers = STORYMEM_MODEL_CONFIG["extra_kwargs"]["num_layers"]
    per_block_modules = [
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "ffn.0",
        "ffn.2",
    ]
    if STORYMEM_MODEL_CONFIG["extra_kwargs"].get("has_image_input"):
        per_block_modules += [
            "cross_attn.k_img",
            "cross_attn.v_img",
        ]
    return [
        f"blocks.{block_id}.{module_name}"
        for block_id in range(num_layers)
        for module_name in per_block_modules
    ]


def print_section(title):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def inspect_safetensors(path, limit, strip_prefix):
    raw_modules = []
    normalized_modules = []
    stripped_prefixes = Counter()

    print_section(f"safetensors keys: {path}")
    with safe_open(path, framework="np") as reader:
        keys = list(reader.keys())
        print(f"total_keys: {len(keys)}")
        for key in keys[:limit]:
            shape = tensor_shape(reader, key)
            shape_text = f" shape={shape}" if shape is not None else ""
            print(f"{key}{shape_text}")

        for key in keys:
            raw_module, _ = parse_lora_module_name(key, strip_prefix=False)
            normalized_module, stripped_prefix = parse_lora_module_name(key, strip_prefix=strip_prefix)
            if raw_module is not None:
                raw_modules.append(raw_module)
            if normalized_module is not None:
                normalized_modules.append(normalized_module)
            if stripped_prefix is not None:
                stripped_prefixes[stripped_prefix] += 1

    raw_modules = sorted(set(raw_modules))
    normalized_modules = sorted(set(normalized_modules))

    print_section("parsed LoRA target modules")
    print(f"raw_module_count: {len(raw_modules)}")
    for name in raw_modules[:limit]:
        print(name)

    if strip_prefix:
        print_section("normalized LoRA target modules")
        print(f"normalized_module_count: {len(normalized_modules)}")
        print(f"stripped_prefixes: {dict(stripped_prefixes)}")
        for name in normalized_modules[:limit]:
            print(name)

    return normalized_modules if strip_prefix else raw_modules


def inspect_storymem_config(limit, show_all):
    print_section("DiffSynth StoryMem model config")
    print(f"model_hash: {STORYMEM_MODEL_CONFIG['model_hash']}")
    print(f"model_name: {STORYMEM_MODEL_CONFIG['model_name']}")
    print(f"model_class: {STORYMEM_MODEL_CONFIG['model_class']}")
    print("extra_kwargs:")
    for key, value in STORYMEM_MODEL_CONFIG["extra_kwargs"].items():
        print(f"  {key}: {value}")

    supported = storymem_supported_modules()
    print_section("DiffSynth StoryMem supported LoRA target module names")
    print(f"supported_module_count: {len(supported)}")
    output = supported if show_all else supported[:limit]
    for name in output:
        print(name)
    if not show_all and len(supported) > limit:
        print(f"... ({len(supported) - limit} more; use --show-all-supported to print all)")
    return supported


def compare_modules(lora_modules, supported_modules, limit):
    supported = set(supported_modules)
    lora = set(lora_modules)
    matched = sorted(lora & supported)
    missing = sorted(lora - supported)

    print_section("LoRA vs DiffSynth StoryMem support")
    print(f"matched_modules: {len(matched)}")
    print(f"unmatched_lora_modules: {len(missing)}")

    if matched:
        print("matched examples:")
        for name in matched[:limit]:
            print(f"  {name}")
    if missing:
        print("unmatched examples:")
        for name in missing[:limit]:
            print(f"  {name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Print safetensors keys and DiffSynth StoryMem/Wan2.2-I2V-A14B "
            "model-config-supported LoRA target module names."
        )
    )
    parser.add_argument("--safetensors", required=True, help="LoRA safetensors file to inspect.")
    parser.add_argument("--limit", type=int, default=80, help="Maximum number of rows per section.")
    parser.add_argument(
        "--strip-wrapper-prefix",
        action="store_true",
        help="Also strip PEFT wrapper prefixes such as base_model.model. before comparing.",
    )
    parser.add_argument(
        "--show-all-supported",
        action="store_true",
        help="Print all supported StoryMem LoRA target module names.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    lora_modules = inspect_safetensors(
        args.safetensors,
        limit=args.limit,
        strip_prefix=args.strip_wrapper_prefix,
    )
    supported_modules = inspect_storymem_config(
        limit=args.limit,
        show_all=args.show_all_supported,
    )
    compare_modules(lora_modules, supported_modules, limit=args.limit)
