"""Diamond Search motion estimation algorithm implementation."""

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


def _get_large_diamond_offsets() -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, -2),
        (2, 0),
        (0, 2),
        (-2, 0),
        (1, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
    ]


def _get_small_diamond_offsets() -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
    ]


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


def _within_search_bounds(
    candidate_x: int,
    candidate_y: int,
    original_x: int,
    original_y: int,
    block_size: int,
    width: int,
    height: int,
    search_range: int,
) -> bool:
    if candidate_x < 0 or candidate_y < 0:
        return False
    if candidate_x + block_size > width or candidate_y + block_size > height:
        return False
    dx = candidate_x - original_x
    dy = candidate_y - original_y
    return abs(dx) <= search_range and abs(dy) <= search_range


def _find_best_match(
    ref_block: np.ndarray,
    target: np.ndarray,
    original_x: int,
    original_y: int,
    search_center_x: int,
    search_center_y: int,
    block_size: int,
    width: int,
    height: int,
    search_range: int,
    offsets: list[tuple[int, int]],
    cost_fn,
) -> tuple[int, int, float, int]:
    best_x = search_center_x
    best_y = search_center_y
    best_cost = float("inf")
    comparisons = 0

    for dx, dy in offsets:
        candidate_x = search_center_x + dx
        candidate_y = search_center_y + dy
        if not _within_search_bounds(
            candidate_x,
            candidate_y,
            original_x,
            original_y,
            block_size,
            width,
            height,
            search_range,
        ):
            continue

        target_block = target[candidate_y : candidate_y + block_size, candidate_x : candidate_x + block_size]
        if target_block.shape != ref_block.shape:
            continue

        cost = cost_fn(ref_block, target_block)
        comparisons += 1
        if cost < best_cost:
            best_cost = cost
            best_x = candidate_x
            best_y = candidate_y

    return best_x, best_y, best_cost, comparisons


def diamond_search_motion_estimation(
    reference_frame: Any,
    target_frame: Any,
    block_size: int = 16,
    search_range: int = 8,
    metric: str = "mad",
    save_path: str | Path | None = None,
    return_stats: bool = False,
) -> list[MotionVector] | tuple[list[MotionVector], dict]:
    """Compute motion vectors using a diamond search algorithm."""
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
            ref_block = reference[y : y + block_size, x : x + block_size]
            search_center_x = x
            search_center_y = y

            if search_range >= 2:
                while True:
                    best_x, best_y, best_cost, comps = _find_best_match(
                        ref_block,
                        target,
                        x,
                        y,
                        search_center_x,
                        search_center_y,
                        block_size,
                        width,
                        height,
                        search_range,
                        _get_large_diamond_offsets(),
                        cost_fn,
                    )
                    comparisons += comps
                    if best_x == search_center_x and best_y == search_center_y:
                        break
                    search_center_x, search_center_y = best_x, best_y

            while True:
                best_x, best_y, best_cost, comps = _find_best_match(
                    ref_block,
                    target,
                    x,
                    y,
                    search_center_x,
                    search_center_y,
                    block_size,
                    width,
                    height,
                    search_range,
                    _get_small_diamond_offsets(),
                    cost_fn,
                )
                comparisons += comps
                if best_x == search_center_x and best_y == search_center_y:
                    break
                search_center_x, search_center_y = best_x, best_y

            motion_vectors.append(
                MotionVector(
                    x=x,
                    y=y,
                    dx=search_center_x - x,
                    dy=search_center_y - y,
                )
            )

    if save_path is not None:
        _save_motion_vectors(motion_vectors, save_path)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = {"comparisons": comparisons, "time_ms": elapsed_ms}
    if return_stats:
        return motion_vectors, stats
    return motion_vectors
