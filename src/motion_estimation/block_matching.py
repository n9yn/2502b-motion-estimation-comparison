"""Block matching metrics for motion estimation."""

import numpy as np


def compute_sad(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """Compute Sum of Absolute Differences for two blocks."""
    # TODO: implement SAD metric
    return float(np.sum(np.abs(block_a.astype(np.float32) - block_b.astype(np.float32))))


def compute_mad(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """Compute Mean Absolute Difference for two blocks."""
    # TODO: implement MAD metric
    return float(np.mean(np.abs(block_a.astype(np.float32) - block_b.astype(np.float32))))


def compute_mse(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """Compute Mean Squared Error for two blocks."""
    # TODO: implement MSE metric
    return float(np.mean((block_a.astype(np.float32) - block_b.astype(np.float32)) ** 2))
