"""Diamond Search motion estimation algorithm."""

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


def _search_candidates(
    reference_block: np.ndarray,
    target: np.ndarray,
    x: int,
    y: int,
    center_dx: int,
    center_dy: int,
    step: int,
    block_size: int,
    search_range: int,
) -> tuple[int, int, int]:
    best_dx = center_dx
    best_dy = center_dy
    best_sad = float("inf")

    for dx, dy in [(0, 0), (-step, 0), (step, 0), (0, -step), (0, step)]:
        candidate_dx = center_dx + dx
        candidate_dy = center_dy + dy
        if abs(candidate_dx) > search_range or abs(candidate_dy) > search_range:
            continue

        candidate_x = x + candidate_dx
        candidate_y = y + candidate_dy
        if (
            candidate_x < 0
            or candidate_y < 0
            or candidate_x + block_size > target.shape[1]
            or candidate_y + block_size > target.shape[0]
        ):
            continue

        target_block = target[candidate_y:candidate_y + block_size, candidate_x:candidate_x + block_size]
        sad = _block_sad(reference_block, target_block)
        if sad < best_sad:
            best_sad = sad
            best_dx = candidate_dx
            best_dy = candidate_dy

    return best_dx, best_dy, best_sad


def diamond_search_motion_estimation(
    reference_frame: Any,
    target_frame: Any,
    block_size: int = 16,
    search_range: int = 8,
) -> list[MotionVector]:
    """Compute motion vectors using a diamond search algorithm."""
    reference = _to_grayscale(reference_frame)
    target = _to_grayscale(target_frame)

    height, width = reference.shape
    motion_vectors: list[MotionVector] = []

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            reference_block = reference[y:y + block_size, x:x + block_size]
            center_dx = 0
            center_dy = 0
            step = max(search_range // 2, 1)

            while step > 0:
                best_dx, best_dy, best_sad = _search_candidates(
                    reference_block,
                    target,
                    x,
                    y,
                    center_dx,
                    center_dy,
                    step,
                    block_size,
                    search_range,
                )

                if best_dx == center_dx and best_dy == center_dy:
                    step //= 2
                else:
                    center_dx, center_dy = best_dx, best_dy

            motion_vectors.append(MotionVector(x, y, center_dx, center_dy))

    print(f"✓ Diamond Search generated {len(motion_vectors)} motion vectors")
    return motion_vectors
