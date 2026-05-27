"""Frame extraction utilities for video preprocessing."""

from pathlib import Path

import cv2


def extract_frames_from_video(video_path: str | Path, output_dir: str | Path) -> None:
    """Extract frames from a video file into the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"Unable to open video file: {video_path}")

    frame_index = 1
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_path = output_path / f"frame_{frame_index:04d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_index += 1

    capture.release()
    print(f"✓ Extracted {frame_index - 1} frames from {video_path} to {output_path}")


def get_frame_path(frame_index: int, output_dir: str | Path) -> Path:
    """Return the expected path for a saved frame image."""
    return Path(output_dir) / f"frame_{frame_index:04d}.png"
