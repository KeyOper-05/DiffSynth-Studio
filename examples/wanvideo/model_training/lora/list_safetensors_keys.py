import argparse
import json

try:
    from safetensors import safe_open
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires safetensors. Install it in the same environment "
        "used for DiffSynth/StoryMem, or run inside that environment."
    ) from exc


LORA_A_MARKERS = (".lora_A.", ".lora_down.")
LORA_B_MARKERS = (".lora_B.", ".lora_up.")


def is_alpha_key(key):
    return key.endswith(".alpha") or key == "alpha" or key.endswith(".lora_alpha")


def tensor_shape(reader, key):
    try:
        return list(reader.get_slice(key).get_shape())
    except Exception:
        return None


def inspect_file(path, show_shapes):
    with safe_open(path, framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        records = []
        for key in keys:
            record = {"key": key}
            if show_shapes:
                record["shape"] = tensor_shape(reader, key)
            records.append(record)

    alpha_keys = [key for key in keys if is_alpha_key(key)]
    lora_a_keys = [key for key in keys if any(marker in key for marker in LORA_A_MARKERS)]
    lora_b_keys = [key for key in keys if any(marker in key for marker in LORA_B_MARKERS)]
    return {
        "path": path,
        "total_keys": len(keys),
        "alpha_key_count": len(alpha_keys),
        "lora_a_key_count": len(lora_a_keys),
        "lora_b_key_count": len(lora_b_keys),
        "alpha_keys": alpha_keys,
        "keys": records,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="List all keys in safetensors files and summarize LoRA alpha keys.")
    parser.add_argument("paths", nargs="+", help="One or more .safetensors files.")
    parser.add_argument("--show_shapes", action="store_true", help="Also print tensor shapes without loading full tensors.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    results = [inspect_file(path, args.show_shapes) for path in args.paths]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    for result in results:
        print(f"File: {result['path']}")
        print(f"Total keys: {result['total_keys']}")
        print(f"LoRA A/down keys: {result['lora_a_key_count']}")
        print(f"LoRA B/up keys: {result['lora_b_key_count']}")
        print(f"Alpha-like keys: {result['alpha_key_count']}")
        if result["alpha_keys"]:
            print("Alpha-like key list:")
            for key in result["alpha_keys"]:
                print(f"  {key}")
        print("All keys:")
        for record in result["keys"]:
            shape = f" shape={record['shape']}" if "shape" in record else ""
            print(f"  {record['key']}{shape}")
        print()


if __name__ == "__main__":
    main()
