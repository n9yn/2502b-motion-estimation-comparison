"""Residual energy analysis utilities."""

import numpy as np


def calculate_residual_energy(residual_frame: np.ndarray) -> float:
    """Calculate the energy of a residual frame."""
    # TODO: compute residual energy using a meaningful norm
    print("[TODO] Calculating residual energy")
    return float(np.sum(np.abs(residual_frame.astype(np.float32))))
