"""Unit tests for residual frame generation and energy analysis."""

from pathlib import Path
import tempfile

import numpy as np

from src.motion_estimation.motion_vector import MotionVector
from src.residual.residual_generator import generate_residual_frame
from src.residual.residual_energy import (
    calculate_residual_energy,
    calculate_residual_error,
    save_residual_statistics,
)


def test_generate_residual_frame_returns_residual() -> None:
    """Residual generator should return a residual frame from motion compensation."""
    reference = np.zeros((16, 16), dtype=np.uint8)
    target = np.zeros((16, 16), dtype=np.uint8)
    reference[2:10, 2:10] = 100
    target[0:8, 0:8] = 100

    motion_vectors = [MotionVector(x=0, y=0, dx=2, dy=2)]
    residual = generate_residual_frame(reference, target, motion_vectors, block_size=8)

    assert residual.shape == reference.shape
    assert isinstance(residual, np.ndarray)
    assert np.all(residual[0:8, 0:8] == 0)


def test_calculate_residual_error_and_energy() -> None:
    """Residual error and energy calculations should be consistent."""
    residual = np.array([[0, -2], [1, -1]], dtype=np.int16)
    errors = calculate_residual_error(residual)
    energy = calculate_residual_energy(residual)

    assert np.isclose(errors["mae"], 1.0)
    assert np.isclose(errors["mse"], 1.5)
    assert np.isclose(errors["rmse"], np.sqrt(1.5))
    assert isinstance(energy, float)
    assert np.isclose(energy, 6.0)


def test_save_residual_statistics_writes_json() -> None:
    """Residual statistics should be saved to a JSON file."""
    residual = np.array([[1, -1], [2, -2]], dtype=np.int16)
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "residual_stats.json"
        stats = save_residual_statistics(residual, output_path)

        assert output_path.exists()
        assert stats["energy"] == calculate_residual_energy(residual)
        assert output_path.read_text(encoding="utf-8")
