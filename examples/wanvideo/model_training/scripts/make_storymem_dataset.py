#!/usr/bin/env python3
"""Build a one-sample StoryMem training dataset from a source video.

The generated dataset matches examples/wanvideo/model_training/train.py:

    metadata.csv columns: video,prompt,memory_images
    video: relative path to a clipped mp4 sample
    memory_images: JSON list string of relative image paths

Example:
    python examples/wanvideo/model_training/scripts/make_storymem_dataset.py \
        --video input.mp4 \
        --start 00:01:23.5 \
        --end 00:01:28.5 \
        --prompt "A character walks into the room." \
        --output data/storymem_single_shot \
        --num-frames 49
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import shutil
import tempfile
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_time(value: str) -> float:
    """Parse seconds or HH:MM:SS(.sss) into seconds."""
    value = str(value).strip()
    if not value:
        raise ValueError("time value must not be empty")
    if ":" not in value:
        seconds = float(value)
    else:
        parts = value.split(":")
        if len(parts) > 3:
            raise ValueError(f"invalid time format: {value}")
        parts = [float(part) for part in parts]
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
    if seconds < 0:
        raise ValueError("time value must be non-negative")
    return seconds


def ensure_num_frames(value: int) -> int:
    """Wan video training expects frame count to be 4n+1."""
    if value <= 0:
        raise ValueError("--num-frames must be positive")
    if value % 4 != 1:
        fixed = value
        while fixed > 1 and fixed % 4 != 1:
            fixed -= 1
        raise ValueError(f"--num-frames must be 4n+1; got {value}. A nearby valid value is {fixed}.")
    return value


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise RuntimeError(f"'{name}' is required but was not found in PATH.")
    return binary


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}\n{message}") from exc


def build_scale_filter(width: int | None, height: int | None) -> str | None:
    if width is None and height is None:
        return None
    width_expr = str(width) if width is not None else "-2"
    height_expr = str(height) if height is not None else "-2"
    return f"scale={width_expr}:{height_expr}:flags=bicubic"


def build_video_filter(fps: float, width: int | None, height: int | None) -> str:
    filters = [f"fps={fps}"]
    scale_filter = build_scale_filter(width, height)
    if scale_filter is not None:
        filters.append(scale_filter)
    return ",".join(filters)


def quality_to_crf(quality: int) -> str:
    return str(max(1, min(31, 33 - quality * 3)))


def evenly_spaced_times(start_seconds: float, end_seconds: float, num_frames: int) -> list[float]:
    if end_seconds <= start_seconds:
        raise ValueError("--end must be greater than --start")
    if num_frames == 1:
        return [start_seconds]
    stride = (end_seconds - start_seconds) / (num_frames - 1)
    return [start_seconds + i * stride for i in range(num_frames)]


def ffmpeg_extract_frame(
    ffmpeg: str,
    source: Path,
    target: Path,
    seconds: float,
    width: int | None,
    height: int | None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{seconds:.6f}",
        "-i",
        str(source),
    ]
    scale_filter = build_scale_filter(width, height)
    if scale_filter is not None:
        command += ["-vf", scale_filter]
    command += ["-frames:v", "1", str(target)]
    run_command(command)


def ffmpeg_encode_frame_sequence(
    ffmpeg: str,
    frame_pattern: Path,
    target: Path,
    fps: float,
    num_frames: int,
    quality: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps}",
        "-i",
        str(frame_pattern),
        "-frames:v",
        str(num_frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        quality_to_crf(quality),
        "-pix_fmt",
        "yuv420p",
        str(target),
    ]
    run_command(command)


def ffmpeg_make_evenly_sampled_video(
    ffmpeg: str,
    source: Path,
    target: Path,
    start_seconds: float,
    end_seconds: float,
    num_frames: int,
    fps: float,
    width: int | None,
    height: int | None,
    quality: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="storymem_frames_") as temp_dir:
        temp_path = Path(temp_dir)
        for index, seconds in enumerate(evenly_spaced_times(start_seconds, end_seconds, num_frames)):
            ffmpeg_extract_frame(
                ffmpeg,
                source,
                temp_path / f"frame_{index:06d}.png",
                seconds,
                width,
                height,
            )
        ffmpeg_encode_frame_sequence(
            ffmpeg,
            temp_path / "frame_%06d.png",
            target,
            fps,
            num_frames,
            quality,
        )


def relative_to_base(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def next_sample_id(metadata_path: Path) -> int:
    if not metadata_path.exists():
        return 0
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def append_metadata(metadata_path: Path, row: dict[str, str]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    exists = metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt", "memory_images"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate StoryMem dataset metadata and video sample.")
    parser.add_argument("--video", required=True, type=Path, help="Source video path.")
    parser.add_argument("--start", required=True, help="Sample start time, in seconds or HH:MM:SS(.sss).")
    parser.add_argument("--end", required=True, help="Sample end time. Frames are sampled evenly from start to end.")
    parser.add_argument("--output", type=Path, default=Path("data/storymem_single_shot"), help="Dataset base directory.")
    parser.add_argument("--metadata-name", default="metadata.csv", help="Metadata filename under --output.")
    parser.add_argument("--prompt", default="", help="Prompt written to metadata.csv.")
    parser.add_argument("--sample-name", default=None, help="Optional sample basename. Defaults to sample_XXXXXX.")
    parser.add_argument("--num-frames", type=int, default=49, help="Number of evenly sampled frames in output sample. Must be 4n+1.")
    parser.add_argument("--fps", type=float, default=None, help="Optional output video fps. Defaults to num_frames / (end - start), preserving playback duration.")
    parser.add_argument("--width", type=int, default=None, help="Optional output video width.")
    parser.add_argument("--height", type=int, default=None, help="Optional output video height.")
    parser.add_argument("--quality", type=int, default=8, help="MP4 quality, usually 1-10.")
    parser.add_argument(
        "--memory-frame",
        action="append",
        default=[],
        type=Path,
        help="Memory image path. Can be provided multiple times.",
    )
    parser.add_argument(
        "--memory-time",
        action="append",
        default=[],
        help="Time in source video to extract as a memory image. Can be provided multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated sample files if --sample-name already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported video extension: {video_path.suffix}")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not 1 <= args.quality <= 10:
        raise ValueError("--quality must be between 1 and 10")

    ffmpeg = require_binary("ffmpeg")
    start_seconds = parse_time(args.start)
    end_seconds = parse_time(args.end)
    if end_seconds <= start_seconds:
        raise ValueError("--end must be greater than --start")
    num_frames = ensure_num_frames(args.num_frames)
    output_fps = args.fps if args.fps is not None else num_frames / (end_seconds - start_seconds)
    output_dir = args.output.expanduser().resolve()
    metadata_path = output_dir / args.metadata_name
    sample_id = next_sample_id(metadata_path)
    sample_name = args.sample_name or f"sample_{sample_id:06d}"

    video_out = output_dir / "videos" / f"{sample_name}.mp4"
    memory_dir = output_dir / "memory" / sample_name
    if not args.overwrite and (video_out.exists() or memory_dir.exists()):
        raise FileExistsError(f"{sample_name} already exists. Use --overwrite or choose --sample-name.")

    ffmpeg_make_evenly_sampled_video(
        ffmpeg, video_path, video_out, start_seconds, end_seconds,
        num_frames, output_fps, args.width, args.height, args.quality,
    )

    if args.overwrite and memory_dir.exists():
        shutil.rmtree(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    memory_paths: list[Path] = []
    for idx, memory_frame in enumerate(args.memory_frame):
        source = memory_frame.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported memory image extension: {source.suffix}")
        target = memory_dir / f"memory_{idx:03d}{source.suffix.lower()}"
        shutil.copyfile(source, target)
        memory_paths.append(target)

    for idx, memory_time in enumerate(args.memory_time, start=len(memory_paths)):
        target = memory_dir / f"memory_{idx:03d}.png"
        ffmpeg_extract_frame(ffmpeg, video_path, target, parse_time(memory_time), args.width, args.height)
        memory_paths.append(target)

    if not memory_paths:
        target = memory_dir / "memory_000.png"
        ffmpeg_extract_frame(ffmpeg, video_out, target, 0.0, None, None)
        memory_paths.append(target)

    row = {
        "video": relative_to_base(video_out, output_dir),
        "prompt": args.prompt,
        "memory_images": json.dumps([relative_to_base(path, output_dir) for path in memory_paths], ensure_ascii=False),
    }
    append_metadata(metadata_path, row)

    print(f"Dataset base: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Video sample: {video_out}")
    print(f"Memory images: {', '.join(str(path) for path in memory_paths)}")
    print("Training flags:")
    print(f"  --dataset_base_path {output_dir}")
    print(f"  --dataset_metadata_path {metadata_path}")
    print("  --data_file_keys video,memory_images")
    print("  --extra_inputs memory_images")


if __name__ == "__main__":
    main()
