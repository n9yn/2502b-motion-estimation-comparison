"""Conversion utilities for image preprocessing."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def convert_to_grayscale(frame: Any) -> Any:
    """Convert a color frame to grayscale."""
    if frame is None:
        raise ValueError("Frame is None")

    array = np.asarray(frame)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    raise ValueError("Unsupported frame format for grayscale conversion")


def batch_convert_directory(input_dir: str | Path, output_dir: str | Path) -> None:
    """Convert all frames in a directory to grayscale and save them."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".png", ".jpg", ".jpeg"}
    frame_files = sorted(
        [path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in supported_extensions]
    )

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue

        gray_frame = convert_to_grayscale(frame)
        output_file = output_path / frame_file.name
        if not cv2.imwrite(str(output_file), gray_frame):
            raise IOError(f"Failed to write grayscale frame: {output_file}")
