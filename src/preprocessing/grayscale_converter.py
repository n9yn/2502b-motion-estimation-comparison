"""Conversion utilities for image preprocessing."""

from pathlib import Path
from typing import Any

import cv2


def convert_to_grayscale(frame: Any) -> Any:
    """Convert a color frame to grayscale."""
    # TODO: implement grayscale conversion using OpenCV
    print("[TODO] Converting frame to grayscale")
    return frame


def batch_convert_directory(input_dir: str | Path, output_dir: str | Path) -> None:
    """Convert all frames in a directory to grayscale and save them."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # TODO: iterate over input frames, convert each to grayscale, and save to output_dir
    print(f"[TODO] Converting frames from {input_dir} to grayscale in {output_dir}")
