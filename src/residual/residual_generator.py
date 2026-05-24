"""Residual frame generation for motion estimation analysis."""

from typing import Any

from src.motion_estimation.motion_vector import MotionVector


def generate_residual_frame(
    reference_frame: Any,
    target_frame: Any,
    motion_vectors: list[MotionVector],
    block_size: int = 16,
) -> Any:
    """Generate a residual frame from reference and target frames."""
    # TODO: construct residual frame using motion compensation
    print("[TODO] Generating residual frame")
    return target_frame
