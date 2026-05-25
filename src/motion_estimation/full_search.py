"""Full Search motion estimation algorithm implementation."""

from pathlib import Path
from typing import Any
import time

import cv2
import numpy as np

from src.motion_estimation.block_matching import compute_mad, compute_mse
from src.motion_estimation.motion_vector import MotionVector


def _normalize_frame(frame: Any) -> np.ndarray:
    if frame is None:
        raise ValueError("Reference and target frames cannot be None")

    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.ndim != 2:
        raise ValueError("Frames must be 2D grayscale or BGR images")
    return array


def _get_cost_function(metric: str):
    metrics = {
        "mad": compute_mad,
        "mse": compute_mse,
    }
    if metric not in metrics:
        raise ValueError(f"Unsupported matching metric: {metric}")
    return metrics[metric]


def _save_motion_vectors(
    motion_vectors: list[MotionVector],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("x,y,dx,dy\n")
        for mv in motion_vectors:
            handle.write(f"{mv.x},{mv.y},{mv.dx},{mv.dy}\n")


def full_search_motion_estimation(
    reference_frame: Any,
    target_frame: Any,
    block_size: int = 16,
    search_range: int = 8,
    metric: str = "mad",
    save_path: str | Path | None = None,
    return_stats: bool = False,
) -> list[MotionVector] | tuple[list[MotionVector], dict]:
    """Compute motion vectors using a full search algorithm."""
    reference = _normalize_frame(reference_frame)
    target = _normalize_frame(target_frame)

    if reference.shape != target.shape:
        raise ValueError("Reference and target frames must have the same shape")

    height, width = reference.shape
    cost_fn = _get_cost_function(metric)
    motion_vectors: list[MotionVector] = []
    comparisons = 0
    t0 = time.perf_counter()

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            reference_block = reference[y : y + block_size, x : x + block_size]
            best_dx = 0
            best_dy = 0
            best_cost = float("inf")

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

                    target_block = target[
                        candidate_y : candidate_y + block_size,
                        candidate_x : candidate_x + block_size,
                    ]
                    comparisons += 1
                    cost = cost_fn(reference_block, target_block)
                    if cost < best_cost:
                        best_cost = cost
                        best_dx = dx
                        best_dy = dy

            motion_vectors.append(MotionVector(x=x, y=y, dx=best_dx, dy=best_dy))

    if save_path is not None:
        _save_motion_vectors(motion_vectors, save_path)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = {"comparisons": comparisons, "time_ms": elapsed_ms}
    if return_stats:
        return motion_vectors, stats
    return motion_vectors
