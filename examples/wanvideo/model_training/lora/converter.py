import argparse
import os

from safetensors.torch import load_file, save_file


PREFIXES_TO_STRIP = (
    "base_model.model.",
    "module.base_model.model.",
)


def convert_key(key):
    for prefix in PREFIXES_TO_STRIP:
        if key.startswith(prefix):
            return key[len(prefix):], True
    return key, False


def convert_lora(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Preset LoRA not found: {src}")

    state_dict = load_file(src, device="cpu")
    converted = {}
    changed = 0
    for key, value in state_dict.items():
        new_key, renamed = convert_key(key)
        changed += int(renamed)
        converted[new_key] = value

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    save_file(converted, dst)
    print(
        "Converted StoryMem preset LoRA for DiffSynth: "
        f"{src} -> {dst} ({changed}/{len(state_dict)} keys renamed)"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert StoryMem PEFT LoRA keys to DiffSynth-compatible keys."
    )
    parser.add_argument("--src", required=True, help="Input StoryMem LoRA safetensors path.")
    parser.add_argument("--dst", required=True, help="Output converted LoRA safetensors path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_lora(args.src, args.dst)
