"""Full Search motion estimation algorithm."""

from typing import Any
from pathlib import Path
import csv
import time

import cv2
import numpy as np

from src.motion_estimation.block_matching import compute_mad, compute_mse, compute_sad
from src.motion_estimation.motion_vector import MotionVector


def _to_grayscale(frame: Any) -> np.ndarray:
    if frame is None:
        raise ValueError("Reference and target frames must not be None")

    if isinstance(frame, np.ndarray):
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    raise ValueError("Unsupported frame type for motion estimation")


def _select_block_metric(metric: str, block_a: np.ndarray, block_b: np.ndarray) -> float:
    metric = metric.lower()
    if metric == "sad":
        return compute_sad(block_a, block_b)
    if metric == "mad":
        return compute_mad(block_a, block_b)
    if metric == "mse":
        return compute_mse(block_a, block_b)
    raise ValueError(f"Unsupported block matching metric: {metric}")


def full_search_motion_estimation(
    reference_frame: Any,
    target_frame: Any,
    block_size: int = 16,
    search_range: int = 8,
    metric: str = "sad",
    return_stats: bool = False,
    save_path: str | Path | None = None,
) -> list[MotionVector] | tuple[list[MotionVector], dict]:
    """Compute motion vectors using a full search algorithm.

    If `return_stats` is True, returns (motion_vectors, stats) where stats contains
    `comparisons` and `time_ms`.
    """
    reference = _to_grayscale(reference_frame)
    target = _to_grayscale(target_frame)

    # Convert once to signed integers to avoid repeated casts during SAD evaluation
    reference_int = reference.astype(np.int16)
    target_int = target.astype(np.int16)

    height, width = reference.shape
    motion_vectors: list[MotionVector] = []

    comparisons = 0
    t0 = time.perf_counter()

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            reference_block = reference_int[y:y + block_size, x:x + block_size]
            best_sad = float("inf")
            best_dx = 0
            best_dy = 0

            min_dx = max(-search_range, -x)
            max_dx = min(search_range, width - block_size - x)
            min_dy = max(-search_range, -y)
            max_dy = min(search_range, height - block_size - y)

            for dy in range(min_dy, max_dy + 1):
                for dx in range(min_dx, max_dx + 1):
                    candidate_x = x + dx
                    candidate_y = y + dy
                    target_block = target_int[candidate_y:candidate_y + block_size, candidate_x:candidate_x + block_size]
                    distance = _select_block_metric(metric, reference_block, target_block)
                    comparisons += 1
                    if distance < best_sad:
                        best_sad = distance
                        best_dx = dx
                        best_dy = dy

            motion_vectors.append(MotionVector(x, y, best_dx, best_dy))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = {"comparisons": comparisons, "time_ms": elapsed_ms}
    print(f"✓ Full Search generated {len(motion_vectors)} motion vectors (comparisons={comparisons}, time={elapsed_ms:.1f}ms)")

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x", "y", "dx", "dy"])
            for mv in motion_vectors:
                writer.writerow(mv.to_tuple())

    if return_stats:
        return motion_vectors, stats
    return motion_vectors
