"""Residual analysis utilities for motion estimation."""

from src.residual.residual_generator import generate_residual_frame
from src.residual.residual_energy import calculate_residual_energy

__all__ = ["generate_residual_frame", "calculate_residual_energy"]
