"""Frame extraction utilities for video preprocessing."""

from pathlib import Path
from typing import Any

import cv2


def extract_frames_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    sample_output_dir: str | Path | None = None,
    sample_frame_count: int = 5,
) -> None:
    """Extract frames from a video file into the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    sample_path = None
    if sample_output_dir is not None:
        sample_path = Path(sample_output_dir)
        sample_path.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_file = get_frame_path(frame_index, output_path)
        if not cv2.imwrite(str(frame_file), frame):
            raise IOError(f"Failed to write frame: {frame_file}")

        if sample_path is not None and frame_index < sample_frame_count:
            sample_frame_file = sample_path / frame_file.name
            if not cv2.imwrite(str(sample_frame_file), frame):
                raise IOError(f"Failed to write sample frame: {sample_frame_file}")

        frame_index += 1

    capture.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames were extracted from {video_path}")

    if not verify_frame_sequence_integrity(output_path):
        raise RuntimeError(f"Frame sequence integrity check failed for {output_path}")


def get_frame_path(frame_index: int, output_dir: str | Path) -> Path:
    """Return the expected path for a saved frame image."""
    return Path(output_dir) / f"frame_{frame_index:04d}.png"


def verify_frame_sequence_integrity(output_dir: str | Path) -> bool:
    """Verify that extracted frames form a contiguous indexed sequence."""
    output_path = Path(output_dir)
    frame_files = sorted(
        [path for path in output_path.iterdir() if path.is_file() and path.suffix.lower() == ".png" and path.name.startswith("frame_")]
    )
    if not frame_files:
        return False

    expected_index = 0
    for frame_file in frame_files:
        stem = frame_file.stem
        try:
            frame_index = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            return False

        if frame_index != expected_index:
            return False
        expected_index += 1

    return True
