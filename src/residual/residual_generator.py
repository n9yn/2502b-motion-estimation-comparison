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


def reconstruct_frame(
    reference_frame: Any,
    motion_vectors: list[MotionVector],
    block_size: int = 16,
) -> np.ndarray:
    """Reconstruct the predicted frame using motion vectors from the reference."""
    reference = _to_grayscale(reference_frame)
    predicted = np.zeros(reference.shape, dtype=np.int16)

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

        block = reference[src_y:src_y + block_size, src_x:src_x + block_size].astype(np.int16)
        predicted[dst_y:dst_y + block_size, dst_x:dst_x + block_size] = block

    return predicted


def generate_residual_frame(
    reference_frame: Any,
    target_frame: Any,
    motion_vectors: list[MotionVector],
    block_size: int = 16,
) -> np.ndarray:
    """Generate a signed residual frame from reference and target frames."""
    target = _to_grayscale(target_frame)
    predicted = reconstruct_frame(reference_frame, motion_vectors, block_size=block_size)
    residual = target.astype(np.int16) - predicted
    return residual
