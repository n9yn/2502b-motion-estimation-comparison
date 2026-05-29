"""Block matching metrics for motion estimation."""

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit
    def _numba_sad(a: np.ndarray, b: np.ndarray) -> float:
        total = 0
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                diff = a[i, j] - b[i, j]
                if diff < 0:
                    diff = -diff
                total += diff
        return total

    @njit
    def _numba_mad(a: np.ndarray, b: np.ndarray) -> float:
        total = 0
        count = a.shape[0] * a.shape[1]
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                diff = a[i, j] - b[i, j]
                if diff < 0:
                    diff = -diff
                total += diff
        return total / count

    @njit
    def _numba_mse(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        count = a.shape[0] * a.shape[1]
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                diff = float(a[i, j]) - float(b[i, j])
                total += diff * diff
        return total / count


def compute_sad(block_a: np.ndarray, block_b: np.ndarray, use_numba: bool = False) -> float:
    """Compute Sum of Absolute Differences for two blocks."""
    if use_numba and NUMBA_AVAILABLE:
        a = block_a.astype(np.int32)
        b = block_b.astype(np.int32)
        return float(_numba_sad(a, b))

    diff = block_a.astype(np.int32) - block_b.astype(np.int32)
    return float(np.abs(diff).sum())


def compute_mad(block_a: np.ndarray, block_b: np.ndarray, use_numba: bool = False) -> float:
    """Compute Mean Absolute Difference for two blocks."""
    if use_numba and NUMBA_AVAILABLE:
        a = block_a.astype(np.int32)
        b = block_b.astype(np.int32)
        return float(_numba_mad(a, b))

    diff = block_a.astype(np.int32) - block_b.astype(np.int32)
    return float(np.mean(np.abs(diff)))


def compute_mse(block_a: np.ndarray, block_b: np.ndarray, use_numba: bool = False) -> float:
    """Compute Mean Squared Error for two blocks."""
    if use_numba and NUMBA_AVAILABLE:
        a = block_a.astype(np.float32)
        b = block_b.astype(np.float32)
        return float(_numba_mse(a, b))

    diff = block_a.astype(np.float32) - block_b.astype(np.float32)
    return float(np.mean(diff * diff))
