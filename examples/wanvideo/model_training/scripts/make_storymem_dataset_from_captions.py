#!/usr/bin/env python3
"""Build a StoryMem MI2V training dataset from ``captions.json``.

Expected input format::

    {
      "v_ApplyEyeMakeup_g01_c01.mp4": {
        "video_path": "/path/to/v_ApplyEyeMakeup_g01_c01.mp4",
        "caption": "In bedroom, woman applies makeup gently."
      }
    }

Each source video is sampled uniformly into a 4n+1-frame MP4.  Its first
output frame is used as both ``input_image`` and the default one-frame
StoryMem memory bank.  The output ``metadata.csv`` is directly compatible
with the StoryMem MI2V cut=False training scripts in this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm"}
METADATA_FIELDS = ["video", "prompt", "memory_images", "input_image", "sample_mode"]


@dataclass(frozen=True)
class CaptionItem:
    source_key: str
    video_path: Path
    caption: str
    sample_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-build a StoryMem MI2V cut=False dataset from captions.json."
    )
    parser.add_argument("captions", type=Path, help="Input captions.json path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/storymem_from_captions"),
        help="Output dataset directory (default: data/storymem_from_captions).",
    )
    parser.add_argument("--metadata-name", default="metadata.csv")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Frames per generated video; must be 4n+1 (default: 49).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="Playback FPS of generated videos (default: 16).",
    )
    parser.add_argument("--width", type=int, default=832, help="Output width (default: 832).")
    parser.add_argument("--height", type=int, default=480, help="Output height (default: 480).")
    parser.add_argument(
        "--quality",
        type=int,
        default=8,
        help="H.264 quality from 1 (small) to 10 (best), default: 8.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files for samples that already exist and rewrite metadata.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned work without creating files.",
    )
    return parser.parse_args()


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"'{name}' is required but was not found in PATH")
    return path


def ensure_num_frames(value: int) -> int:
    if value <= 1 or value % 4 != 1:
        raise ValueError(f"--num-frames must be greater than 1 and equal to 4n+1; got {value}")
    return value


def sanitize_sample_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(value).stem).strip("._")
    return name or "sample"


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def normalize_records(data: Any) -> list[tuple[str, Any]]:
    if isinstance(data, dict):
        return [(str(key), value) for key, value in data.items()]
    if isinstance(data, list):
        records = []
        for index, value in enumerate(data):
            key = value.get("name", f"sample_{index:06d}") if isinstance(value, dict) else f"sample_{index:06d}"
            records.append((str(key), value))
        return records
    raise ValueError("captions.json must contain a JSON object (or a list of record objects)")


def load_caption_items(captions_path: Path) -> list[CaptionItem]:
    records = normalize_records(load_json(captions_path))
    if not records:
        raise ValueError("captions.json contains no samples")

    parsed: list[tuple[str, Path, str, str]] = []
    name_counts: dict[str, int] = {}
    errors: list[str] = []
    for source_key, record in records:
        if not isinstance(record, dict):
            errors.append(f"{source_key!r}: value must be an object")
            continue
        video_value = record.get("video_path")
        caption = record.get("caption")
        if not isinstance(video_value, str) or not video_value.strip():
            errors.append(f"{source_key!r}: missing non-empty 'video_path'")
            continue
        if not isinstance(caption, str) or not caption.strip():
            errors.append(f"{source_key!r}: missing non-empty 'caption'")
            continue

        video_path = Path(video_value).expanduser()
        if not video_path.is_absolute():
            video_path = captions_path.parent / video_path
        video_path = video_path.resolve()
        if not video_path.is_file():
            errors.append(f"{source_key!r}: video does not exist: {video_path}")
            continue
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            errors.append(f"{source_key!r}: unsupported video extension: {video_path.suffix}")
            continue

        base_name = sanitize_sample_name(source_key)
        name_counts[base_name] = name_counts.get(base_name, 0) + 1
        parsed.append((source_key, video_path, caption.strip(), base_name))

    if errors:
        preview = "\n".join(f"  - {message}" for message in errors[:20])
        suffix = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"captions.json validation failed:\n{preview}{suffix}")

    items = []
    for source_key, video_path, caption, base_name in parsed:
        sample_name = base_name
        if name_counts[base_name] > 1:
            digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8]
            sample_name = f"{base_name}_{digest}"
        items.append(CaptionItem(source_key, video_path, caption, sample_name))
    return items


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}\n{detail}") from exc


def probe_duration(ffprobe: str, video_path: Path) -> float:
    command = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"could not determine video duration: {video_path}") from exc
    if duration <= 0:
        raise RuntimeError(f"video duration must be positive: {video_path}")
    return duration


def probe_frame_count(ffprobe: str, video_path: Path) -> int:
    command = [
        ffprobe,
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"could not count frames in generated video: {video_path}") from exc


def quality_to_crf(quality: int) -> str:
    return str(max(1, min(31, 33 - quality * 3)))


def make_video(
    ffmpeg: str,
    source: Path,
    target: Path,
    duration: float,
    num_frames: int,
    fps: float,
    width: int,
    height: int,
    quality: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Sampling FPS covers [0, duration) uniformly and produces exactly num_frames.
    sample_fps = num_frames / duration
    video_filter = (
        "tpad=stop_mode=clone:stop_duration=1,"
        f"fps={sample_fps:.12f},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=bicubic,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,setpts=N/({fps:.12f}*TB)"
    )
    command = [
        ffmpeg, "-y", "-i", str(source),
        "-map", "0:v:0", "-an", "-vf", video_filter,
        "-frames:v", str(num_frames), "-r", f"{fps:.12f}", "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "medium", "-crf", quality_to_crf(quality),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ]
    run_command(command)


def extract_first_frame(ffmpeg: str, video_path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run_command([ffmpeg, "-y", "-i", str(video_path), "-map", "0:v:0", "-frames:v", "1", str(target)])


def relative_to_base(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    captions_path = args.captions.expanduser().resolve()
    if not captions_path.is_file():
        raise FileNotFoundError(captions_path)
    num_frames = ensure_num_frames(args.num_frames)
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise ValueError("--width and --height must be positive even integers")
    if not 1 <= args.quality <= 10:
        raise ValueError("--quality must be between 1 and 10")

    items = load_caption_items(captions_path)
    output_dir = args.output.expanduser().resolve()
    metadata_path = output_dir / args.metadata_name
    print(f"Validated {len(items)} samples from {captions_path}")
    print(f"Output: {output_dir}")
    if args.dry_run:
        for item in items[:10]:
            print(f"  {item.source_key} -> videos/{item.sample_name}.mp4")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
        return 0

    ffmpeg = require_binary("ffmpeg")
    ffprobe = require_binary("ffprobe")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        video_out = output_dir / "videos" / f"{item.sample_name}.mp4"
        input_image = output_dir / "input_images" / item.sample_name / "first_frame.png"
        memory_image = output_dir / "memory" / item.sample_name / "memory_000.png"
        existing = [path for path in (video_out, input_image, memory_image) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"sample {item.sample_name!r} already exists ({existing[0]}). "
                "Use --overwrite to replace generated files."
            )

        print(f"[{index}/{len(items)}] {item.source_key}")
        duration = probe_duration(ffprobe, item.video_path)
        make_video(
            ffmpeg, item.video_path, video_out, duration, num_frames,
            args.fps, args.width, args.height, args.quality,
        )
        actual_frames = probe_frame_count(ffprobe, video_out)
        if actual_frames != num_frames:
            raise RuntimeError(
                f"generated {actual_frames} frames instead of {num_frames}: {video_out}"
            )
        extract_first_frame(ffmpeg, video_out, input_image)
        memory_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_image, memory_image)
        rows.append(
            {
                "video": relative_to_base(video_out, output_dir),
                "prompt": item.caption,
                "memory_images": json.dumps(
                    [relative_to_base(memory_image, output_dir)], ensure_ascii=False
                ),
                "input_image": relative_to_base(input_image, output_dir),
                "sample_mode": "mi2v_cut_false",
            }
        )

    write_metadata(metadata_path, rows)
    print(f"Done: {len(rows)} samples")
    print(f"Metadata: {metadata_path}")
    print("Training flags:")
    print(f"  --dataset_base_path {output_dir}")
    print(f"  --dataset_metadata_path {metadata_path}")
    print("  --data_file_keys video,memory_images,input_image")
    print("  --extra_inputs memory_images,input_image")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
