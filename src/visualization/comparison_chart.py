"""Comparison chart generation for algorithm analysis."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_runtime_comparison(data: dict[str, float], output_path: str | Path) -> None:
    """Generate a runtime comparison chart for motion estimation algorithms."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # TODO: create runtime and energy comparison charts using matplotlib
    print(f"[TODO] Plotting runtime comparison to {output_path}")


def plot_energy_comparison(data: dict[str, float], output_path: str | Path) -> None:
    """Generate an energy comparison chart for residual analysis."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # TODO: create residual energy comparison plots using matplotlib
    print(f"[TODO] Plotting energy comparison to {output_path}")
