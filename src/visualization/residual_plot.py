"""Residual frame plotting utilities."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_residual_frame(frame: Any, output_path: str | Path) -> None:
    """Plot and save a residual frame image."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # TODO: implement grayscale residual frame plotting and saving
    print(f"[TODO] Plotting residual frame to {output_path}")
