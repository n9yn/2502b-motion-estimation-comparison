"""Unit tests for residual frame generation and energy analysis."""

from src.residual.residual_generator import generate_residual_frame
from src.residual.residual_energy import calculate_residual_energy


def test_generate_residual_frame_returns_frame() -> None:
    """Residual generator should return a frame object."""
    result = generate_residual_frame(None, None, [])
    assert result is not None


def test_calculate_residual_energy_returns_float() -> None:
    """Residual energy calculation should return a float value."""
    energy = calculate_residual_energy(__import__("numpy").zeros((1, 1), dtype="uint8"))
    assert isinstance(energy, float)
