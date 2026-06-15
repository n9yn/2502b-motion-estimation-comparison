"""Performance metrics for compression and motion estimation."""

import numpy as np


def compute_mse(reference: np.ndarray, target: np.ndarray) -> float:
    """Compute mean squared error between reference and target frames."""
    if reference.shape != target.shape:
        raise ValueError("Reference and target arrays must have the same shape.")

    difference = reference.astype(np.float32) - target.astype(np.float32)
    return float(np.mean(np.square(difference)))


def compute_psnr(reference: np.ndarray, target: np.ndarray, max_pixel_value: float = 255.0) -> float:
    """Compute PSNR between two frames."""
    mse = compute_mse(reference, target)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(max_pixel_value / np.sqrt(mse))


def compute_mse_from_residual(residual: np.ndarray) -> float:
    """Compute MSE directly from a residual (error) frame."""
    residual_float = residual.astype(np.float32)
    return float(np.mean(np.square(residual_float)))


def compute_psnr_from_residual(residual: np.ndarray, max_pixel_value: float = 255.0) -> float:
    """Compute PSNR from a residual frame representing target minus predicted."""
    mse = compute_mse_from_residual(residual)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(max_pixel_value / np.sqrt(mse))
