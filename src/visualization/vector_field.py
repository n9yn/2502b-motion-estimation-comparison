"""Visualization for motion vector fields."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.motion_estimation.motion_vector import MotionVector


def plot_motion_vectors(
    frame: Any,
    motion_vectors: list[MotionVector],
    output_path: str | Path,
) -> None:
    """Plot motion vectors on top of a reference frame."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # TODO: implement vector field plotting using matplotlib quiver or arrow plots
    print(f"[TODO] Plotting motion vectors to {output_path}")
