"""Residual energy analysis utilities."""

import json
from pathlib import Path

import numpy as np


def calculate_residual_error(residual_frame: np.ndarray) -> dict[str, float]:
    """Calculate residual error metrics for a residual frame."""
    residual = np.asarray(residual_frame, dtype=np.float32)
    absolute = np.abs(residual)
    mse = float(np.mean(np.square(absolute)))
    return {
        "mae": float(np.mean(absolute)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "max_abs": float(np.max(absolute)),
        "min_abs": float(np.min(absolute)),
        "mean_abs": float(np.mean(absolute)),
        "std_abs": float(np.std(absolute)),
    }


def calculate_residual_energy(residual_frame: np.ndarray) -> float:
    """Calculate the energy of a residual frame."""
    residual = np.asarray(residual_frame, dtype=np.float32)
    return float(np.sum(np.square(residual)))


def save_residual_statistics(
    residual_frame: np.ndarray,
    output_path: str | Path,
) -> dict[str, float]:
    """Save residual metrics and energy statistics to a JSON file."""
    statistics = calculate_residual_error(residual_frame)
    statistics["energy"] = calculate_residual_energy(residual_frame)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2)

    return statistics
