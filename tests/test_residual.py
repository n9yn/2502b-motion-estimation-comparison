"""Unit tests for residual frame generation and energy analysis."""

import numpy as np
from src.motion_estimation.motion_vector import MotionVector
from src.residual.residual_generator import generate_residual_frame
from src.residual.residual_energy import calculate_residual_energy


def test_generate_residual_frame_returns_frame() -> None:
    """Residual generator should return a frame object."""
    reference = np.zeros((32, 32), dtype=np.uint8)
    target = np.zeros((32, 32), dtype=np.uint8)
    target[4:20, 4:20] = 50
    motion_vectors = [MotionVector(0, 0, 4, 4)]

    result = generate_residual_frame(reference, target, motion_vectors, block_size=16)
    assert result.shape == target.shape
    assert isinstance(result, np.ndarray)


def test_calculate_residual_energy_returns_float() -> None:
    """Residual energy calculation should return a float value."""
    energy = calculate_residual_energy(np.zeros((1, 1), dtype=np.uint8))
    assert isinstance(energy, float)
