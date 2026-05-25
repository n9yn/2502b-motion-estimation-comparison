"""Residual frame generation for motion estimation analysis."""

from typing import Any

import cv2
import numpy as np

from src.motion_estimation.motion_vector import MotionVector


def _to_grayscale(frame: Any) -> np.ndarray:
    if frame is None:
        raise ValueError("Reference and target frames must not be None")

    if isinstance(frame, np.ndarray):
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    raise ValueError("Unsupported frame type for residual generation")


def generate_residual_frame(
    reference_frame: Any,
    target_frame: Any,
    motion_vectors: list[MotionVector],
    block_size: int = 16,
) -> Any:
    """Generate a residual frame from reference and target frames."""
    reference = _to_grayscale(reference_frame)
    target = _to_grayscale(target_frame)
    residual = np.zeros_like(target)
    predicted = np.zeros_like(target)

    for vector in motion_vectors:
        src_x = vector.x
        src_y = vector.y
        dst_x = src_x + vector.dx
        dst_y = src_y + vector.dy

        if (
            src_x < 0
            or src_y < 0
            or dst_x < 0
            or dst_y < 0
            or src_x + block_size > reference.shape[1]
            or src_y + block_size > reference.shape[0]
            or dst_x + block_size > reference.shape[1]
            or dst_y + block_size > reference.shape[0]
        ):
            continue

        predicted[dst_y:dst_y + block_size, dst_x:dst_x + block_size] = reference[src_y:src_y + block_size, src_x:src_x + block_size]

    residual = cv2.absdiff(target, predicted)
    return residual
