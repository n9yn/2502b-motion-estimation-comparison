"""Residual frame generation for motion estimation analysis."""

from typing import Any

import numpy as np

from src.motion_estimation.motion_vector import MotionVector


def _normalize_frame(frame: Any) -> np.ndarray:
    if frame is None:
        raise ValueError("Reference and target frames cannot be None")

    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] == 3:
        import cv2

        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    if array.ndim != 2:
        raise ValueError("Frames must be 2D grayscale or BGR images")

    return array


def _build_predicted_frame(
    reference_frame: np.ndarray,
    motion_vectors: list[MotionVector],
    block_size: int,
) -> np.ndarray:
    height, width = reference_frame.shape
    predicted = np.zeros_like(reference_frame)

    for mv in motion_vectors:
        src_x = mv.x + mv.dx
        src_y = mv.y + mv.dy
        dst_x = mv.x
        dst_y = mv.y

        if (
            src_x < 0
            or src_y < 0
            or dst_x < 0
            or dst_y < 0
            or src_x + block_size > width
            or src_y + block_size > height
            or dst_x + block_size > width
            or dst_y + block_size > height
        ):
            continue

        predicted[dst_y : dst_y + block_size, dst_x : dst_x + block_size] = reference_frame[
            src_y : src_y + block_size, src_x : src_x + block_size
        ]

    return predicted


def generate_residual_frame(
    reference_frame: Any,
    target_frame: Any,
    motion_vectors: list[MotionVector],
    block_size: int = 16,
) -> np.ndarray:
    """Generate a residual frame from reference and target frames using motion compensation."""
    reference = _normalize_frame(reference_frame)
    target = _normalize_frame(target_frame)

    if reference.shape != target.shape:
        raise ValueError("Reference and target frames must have the same shape")

    predicted = _build_predicted_frame(reference, motion_vectors, block_size)
    residual = target.astype(np.int16) - predicted.astype(np.int16)

    return residual
