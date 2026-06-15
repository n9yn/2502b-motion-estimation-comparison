"""Conversion utilities for image preprocessing."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def convert_to_grayscale(frame: Any) -> Any:
    """Convert a color frame to grayscale."""
    if frame is None:
        return frame

    if isinstance(frame, np.ndarray):
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


def batch_convert_directory(input_dir: str | Path, output_dir: str | Path) -> None:
    """Convert all frames in a directory to grayscale and save them."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for frame_path in sorted(input_path.glob("*.png")):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        gray_frame = convert_to_grayscale(frame)
        output_file = output_path / frame_path.name
        cv2.imwrite(str(output_file), gray_frame)

    print(f"✓ Converted {len(list(output_path.glob('*.png')))} frames from {input_path} to grayscale in {output_path}")
