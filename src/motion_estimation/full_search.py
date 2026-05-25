"""Full Search motion estimation algorithm."""

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

    raise ValueError("Unsupported frame type for motion estimation")


def _block_sad(block_a: np.ndarray, block_b: np.ndarray) -> int:
    return int(np.sum(np.abs(block_a.astype(np.int16) - block_b.astype(np.int16))))


def full_search_motion_estimation(
    reference_frame: Any,
    target_frame: Any,
    block_size: int = 16,
    search_range: int = 8,
) -> list[MotionVector]:
    """Compute motion vectors using a full search algorithm."""
    reference = _to_grayscale(reference_frame)
    target = _to_grayscale(target_frame)

    height, width = reference.shape
    motion_vectors: list[MotionVector] = []

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            reference_block = reference[y:y + block_size, x:x + block_size]
            best_sad = float("inf")
            best_dx = 0
            best_dy = 0

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    candidate_x = x + dx
                    candidate_y = y + dy
                    if (
                        candidate_x < 0
                        or candidate_y < 0
                        or candidate_x + block_size > width
                        or candidate_y + block_size > height
                    ):
                        continue

                    target_block = target[candidate_y:candidate_y + block_size, candidate_x:candidate_x + block_size]
                    sad = _block_sad(reference_block, target_block)
                    if sad < best_sad:
                        best_sad = sad
                        best_dx = dx
                        best_dy = dy

            motion_vectors.append(MotionVector(x, y, best_dx, best_dy))

    print(f"✓ Full Search generated {len(motion_vectors)} motion vectors")
    return motion_vectors
