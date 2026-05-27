"""Block matching metrics for motion estimation."""

import numpy as np


def compute_sad(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """Compute Sum of Absolute Differences for two blocks."""
    # Use integer arithmetic when possible for speed
    diff = block_a.astype(np.int32) - block_b.astype(np.int32)
    return float(np.abs(diff).sum())


def compute_mad(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """Compute Mean Absolute Difference for two blocks."""
    diff = block_a.astype(np.int32) - block_b.astype(np.int32)
    return float(np.mean(np.abs(diff)))


def compute_mse(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """Compute Mean Squared Error for two blocks."""
    diff = block_a.astype(np.float32) - block_b.astype(np.float32)
    return float(np.mean(diff * diff))
