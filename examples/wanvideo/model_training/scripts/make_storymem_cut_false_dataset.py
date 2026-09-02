#!/usr/bin/env python3
"""Build a one-sample cut=False StoryMem MI2V training dataset from one video.

The output video is constructed as:

    memory images: sampled only from [--memory-start, --prev-end)
    frame 0: previous shot last frame, sampled at --prev-end
    frame 1..N-1: current shot frames, sampled strictly after --prev-end

The generated metadata matches the DiffSynth StoryMem training path with:

    --data_file_keys video,memory_images,input_image
    --extra_inputs memory_images,input_image
"""

from __future__ import annotations

import argparse
import csv
import fractions
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
METADATA_FIELDS = ["video", "prompt", "memory_images", "input_image", "sample_mode"]


def parse_time(value: str) -> float:
    value = str(value).strip()
    if not value:
        raise ValueError("time value must not be empty")
    if ":" not in value:
        seconds = float(value)
    else:
        parts = value.split(":")
        if len(parts) > 3:
            raise ValueError(f"invalid time format: {value}")
        seconds = 0.0
        for part in [float(part) for part in parts]:
            seconds = seconds * 60 + part
    if seconds < 0:
        raise ValueError("time value must be non-negative")
    return seconds


def ensure_num_frames(value: int) -> int:
    if value <= 1:
        raise ValueError("--num-frames must be greater than 1")
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


def probe_video_fps(ffprobe: str, source: Path) -> float:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(source),
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}\n{message}") from exc
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value or value == "0/0":
            continue
        fps = float(fractions.Fraction(value))
        if fps > 0:
            return fps
    raise RuntimeError(f"could not determine video fps for {source}")


def build_scale_filter(width: int | None, height: int | None) -> str | None:
    if width is None and height is None:
        return None
    width_expr = str(width) if width is not None else "-2"
    height_expr = str(height) if height is not None else "-2"
    return f"scale={width_expr}:{height_expr}:flags=bicubic"


def quality_to_crf(quality: int) -> str:
    return str(max(1, min(31, 33 - quality * 3)))


def evenly_spaced_times(start_seconds: float, end_seconds: float, num_frames: int) -> list[float]:
    if end_seconds <= start_seconds:
        raise ValueError("--end must be greater than --start")
    if num_frames == 1:
        return [start_seconds]
    stride = (end_seconds - start_seconds) / (num_frames - 1)
    return [start_seconds + i * stride for i in range(num_frames)]


def interval_center_times(start_seconds: float, end_seconds: float, count: int) -> list[float]:
    """Sample deterministic timestamps strictly inside a half-open memory interval."""
    if end_seconds <= start_seconds:
        raise ValueError("--prev-end must be greater than --memory-start")
    if count <= 0:
        raise ValueError("--num-memory-images must be positive")
    stride = (end_seconds - start_seconds) / count
    return [start_seconds + (index + 0.5) * stride for index in range(count)]


def validate_memory_times(times: list[float], start_seconds: float, end_seconds: float) -> None:
    for seconds in times:
        if not start_seconds <= seconds < end_seconds:
            raise ValueError(
                f"--memory-time {seconds:.6f} is outside the memory-only interval "
                f"[{start_seconds:.6f}, {end_seconds:.6f})."
            )


def validate_continuous_timeline(
    memory_start_seconds: float,
    prev_end_seconds: float,
    start_seconds: float,
    end_seconds: float,
    source_fps: float,
) -> float:
    """Validate a cut=False timeline and return the first supervised-frame time."""
    if source_fps <= 0:
        raise ValueError("source fps must be positive")
    if prev_end_seconds <= memory_start_seconds:
        raise ValueError("--prev-end must be greater than --memory-start")
    frame_duration = 1.0 / source_fps
    if abs(start_seconds - prev_end_seconds) > frame_duration / 2:
        raise ValueError(
            "cut=False MI2V requires --start and --prev-end to identify the same boundary "
            "frame (difference must be at most half a source-video frame)."
        )
    current_start_seconds = prev_end_seconds + frame_duration
    if current_start_seconds >= end_seconds:
        raise ValueError("--end must be after the first source-video frame following --prev-end")
    return current_start_seconds


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


def ffmpeg_make_cut_false_video(
    ffmpeg: str,
    source: Path,
    previous_last_frame: Path,
    target: Path,
    start_seconds: float,
    end_seconds: float,
    num_frames: int,
    fps: float,
    width: int | None,
    height: int | None,
    quality: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="storymem_cut_false_frames_") as temp_dir:
        temp_path = Path(temp_dir)
        shutil.copyfile(previous_last_frame, temp_path / "frame_000000.png")
        current_frame_count = num_frames - 1
        for index, seconds in enumerate(evenly_spaced_times(start_seconds, end_seconds, current_frame_count), start=1):
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
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a cut=False StoryMem MI2V dataset sample from one source video.")
    parser.add_argument("--video", required=True, type=Path, help="Source video containing previous-shot end and current-shot frames.")
    parser.add_argument("--memory-start", required=True, help="Start of the memory-only source interval.")
    parser.add_argument("--prev-end", required=True, help="End of the memory-only interval and time of the MI2V boundary frame.")
    parser.add_argument("--start", required=True, help="Current-shot boundary time; for cut=False this must match --prev-end.")
    parser.add_argument("--end", required=True, help="Current shot end time.")
    parser.add_argument("--output", type=Path, default=Path("data/storymem_cut_false_mi2v"), help="Dataset base directory.")
    parser.add_argument("--metadata-name", default="metadata.csv", help="Metadata filename under --output.")
    parser.add_argument("--prompt", required=True, help="Prompt written to metadata.csv.")
    parser.add_argument("--sample-name", default=None, help="Optional sample basename. Defaults to sample_XXXXXX.")
    parser.add_argument("--num-frames", type=int, default=49, help="Final output frame count. Must be 4n+1.")
    parser.add_argument("--fps", type=float, default=None, help="Output video fps. Defaults to num_frames / (end - start).")
    parser.add_argument("--width", type=int, default=None, help="Optional output video width.")
    parser.add_argument("--height", type=int, default=None, help="Optional output video height.")
    parser.add_argument("--quality", type=int, default=8, help="MP4 quality, usually 1-10.")
    parser.add_argument(
        "--num-memory-images",
        type=int,
        default=3,
        help="Number of images sampled uniformly inside [--memory-start, --prev-end) when no explicit memory is supplied.",
    )
    parser.add_argument(
        "--memory-frame",
        action="append",
        default=[],
        type=Path,
        help="External historical memory image path. Can be repeated; the caller must ensure it is not from the target shot.",
    )
    parser.add_argument(
        "--memory-time",
        action="append",
        default=[],
        help="Time in the memory-only interval to extract as a memory image. Can be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated sample files if --sample-name already exists.")
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

    memory_start_seconds = parse_time(args.memory_start)
    prev_end_seconds = parse_time(args.prev_end)
    start_seconds = parse_time(args.start)
    end_seconds = parse_time(args.end)
    if args.num_memory_images <= 0:
        raise ValueError("--num-memory-images must be positive")

    ffmpeg = require_binary("ffmpeg")
    ffprobe = require_binary("ffprobe")
    source_fps = probe_video_fps(ffprobe, video_path)
    current_start_seconds = validate_continuous_timeline(
        memory_start_seconds,
        prev_end_seconds,
        start_seconds,
        end_seconds,
        source_fps,
    )
    explicit_memory_times = [parse_time(value) for value in args.memory_time]
    validate_memory_times(explicit_memory_times, memory_start_seconds, prev_end_seconds)
    explicit_memory_frames = [path.expanduser().resolve() for path in args.memory_frame]
    for source in explicit_memory_frames:
        if not source.exists():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported memory image extension: {source.suffix}")
    num_frames = ensure_num_frames(args.num_frames)
    output_fps = args.fps if args.fps is not None else num_frames / (end_seconds - start_seconds)
    output_dir = args.output.expanduser().resolve()
    metadata_path = output_dir / args.metadata_name
    sample_id = next_sample_id(metadata_path)
    sample_name = args.sample_name or f"sample_{sample_id:06d}"

    video_out = output_dir / "videos" / f"{sample_name}.mp4"
    input_image_dir = output_dir / "input_images" / sample_name
    input_image = input_image_dir / "previous_last_frame.png"
    memory_dir = output_dir / "memory" / sample_name
    existing_paths = [video_out, input_image_dir, memory_dir]
    if not args.overwrite and any(path.exists() for path in existing_paths):
        raise FileExistsError(f"{sample_name} already exists. Use --overwrite or choose --sample-name.")
    if args.overwrite:
        for path in (input_image_dir, memory_dir):
            if path.exists():
                shutil.rmtree(path)

    input_image_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_extract_frame(ffmpeg, video_path, input_image, prev_end_seconds, args.width, args.height)
    ffmpeg_make_cut_false_video(
        ffmpeg,
        video_path,
        input_image,
        video_out,
        current_start_seconds,
        end_seconds,
        num_frames,
        output_fps,
        args.width,
        args.height,
        args.quality,
    )

    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_paths: list[Path] = []
    for idx, source in enumerate(explicit_memory_frames):
        target = memory_dir / f"memory_{idx:03d}{source.suffix.lower()}"
        shutil.copyfile(source, target)
        memory_paths.append(target)

    for idx, seconds in enumerate(explicit_memory_times, start=len(memory_paths)):
        target = memory_dir / f"memory_{idx:03d}.png"
        ffmpeg_extract_frame(ffmpeg, video_path, target, seconds, args.width, args.height)
        memory_paths.append(target)

    if not memory_paths:
        memory_times = interval_center_times(
            memory_start_seconds,
            prev_end_seconds,
            args.num_memory_images,
        )
        for idx, seconds in enumerate(memory_times):
            target = memory_dir / f"memory_{idx:03d}.png"
            ffmpeg_extract_frame(ffmpeg, video_path, target, seconds, args.width, args.height)
            memory_paths.append(target)

    row = {
        "video": relative_to_base(video_out, output_dir),
        "prompt": args.prompt,
        "memory_images": json.dumps([relative_to_base(path, output_dir) for path in memory_paths], ensure_ascii=False),
        "input_image": relative_to_base(input_image, output_dir),
        "sample_mode": "mi2v_cut_false",
    }
    append_metadata(metadata_path, row)

    print(f"Dataset base: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Video sample: {video_out}")
    print(f"Input image: {input_image}")
    print(f"Memory images: {', '.join(str(path) for path in memory_paths)}")
    print(f"Memory-only interval: [{memory_start_seconds:.6f}, {prev_end_seconds:.6f})")
    print(f"Current shot sampling starts at: {current_start_seconds:.6f}s")
    print("Training flags:")
    print(f"  --dataset_base_path {output_dir}")
    print(f"  --dataset_metadata_path {metadata_path}")
    print("  --data_file_keys video,memory_images,input_image")
    print("  --extra_inputs memory_images,input_image")


if __name__ == "__main__":
    main()
