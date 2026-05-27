"""Block matching metrics for motion estimation."""

import numpy as np

# Try to import numba for JIT acceleration; gracefully fallback if unavailable
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba not available
    def njit(func):
        return func


@njit
def _sad_numba(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """Numba-accelerated Sum of Absolute Differences."""
    result = 0.0
    for i in range(block_a.shape[0]):
        for j in range(block_a.shape[1]):
            result += abs(float(block_a[i, j]) - float(block_b[i, j]))
    return result


def compute_sad(block_a: np.ndarray, block_b: np.ndarray, use_numba: bool = False) -> float:
    """Compute Sum of Absolute Differences for two blocks.
    
    Args:
        block_a: First block
        block_b: Second block
        use_numba: If True and numba is available, use JIT acceleration
    
    Returns:
        SAD value
    """
    if use_numba and HAS_NUMBA:
        return _sad_numba(block_a.astype(np.float32), block_b.astype(np.float32))
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
