from pathlib import Path
import tempfile
import time
from typing import Tuple

import cv2


def save_uploaded_temp(uploaded_file) -> Path:
    """Save a Streamlit uploaded file to a temporary path and return it."""
    suffix = Path(uploaded_file.name).suffix
    tmp = Path(tempfile.gettempdir()) / f"uploaded_video_{int(time.time())}{suffix}"
    with open(tmp, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return tmp


def get_video_metadata(video_path: Path) -> dict:
    """Return basic metadata for a video: width, height, fps, frame_count, duration."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": float(fps),
        "frame_count": frame_count,
        "duration": float(duration),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
