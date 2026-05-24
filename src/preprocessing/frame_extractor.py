"""Frame extraction utilities for video preprocessing."""

from pathlib import Path
from typing import Optional

import cv2


def extract_frames_from_video(video_path: str | Path, output_dir: str | Path) -> None:
    """Extract frames from a video file into the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # TODO: implement actual frame extraction using OpenCV
    #       Save extracted frames into output_path.
    print(f"[TODO] Extracting frames from {video_path} into {output_path}")


def get_frame_path(frame_index: int, output_dir: str | Path) -> Path:
    """Return the expected path for a saved frame image."""
    return Path(output_dir) / f"frame_{frame_index:04d}.png"
