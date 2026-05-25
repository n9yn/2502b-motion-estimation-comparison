"""Utilities to extract residual energy series and save comparison charts/images."""

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.motion_estimation.full_search import full_search_motion_estimation
from src.motion_estimation.diamond_search import diamond_search_motion_estimation
from src.residual.residual_generator import generate_residual_frame
from src.residual.residual_energy import calculate_residual_energy
from src.visualization.comparison_chart import plot_residual_energy_series
from src.visualization.visualization_manager import VisualizationManager


def extract_residual_energy_series(
    frames_dir: str | Path,
    output_dir: str | Path,
    block_size: int = 16,
    search_range: int = 8,
) -> Dict[str, List[float]]:
    """Compute residual energy series for Full Search and Diamond Search.

    This function processes consecutive frame pairs in `frames_dir`, computes motion
    vectors using both algorithms, generates residual frames, computes their energy,
    and saves per-frame residual visualizations and a final energy series chart.

    Returns a dict mapping algorithm names to list of energies.
    """
    frames_path = Path(frames_dir)
    out_path = Path(output_dir)
    vis = VisualizationManager(out_path / "visualizations")
    charts_dir = out_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_path.glob("*.png"))
    if len(frame_files) < 2:
        raise ValueError("Need at least two frames to compute residual series")

    energies_fs: List[float] = []
    energies_diamond: List[float] = []

    for idx in range(len(frame_files) - 1):
        f1 = cv2.imread(str(frame_files[idx]), cv2.IMREAD_COLOR)
        f2 = cv2.imread(str(frame_files[idx + 1]), cv2.IMREAD_COLOR)
        if f1 is None or f2 is None:
            continue

        # Convert to grayscale for estimation and residual computation
        ref_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        tar_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        # Compute motion vectors
        vectors_fs = full_search_motion_estimation(ref_gray, tar_gray, block_size, search_range)
        vectors_diamond = diamond_search_motion_estimation(ref_gray, tar_gray, block_size, search_range)

        # Generate residuals
        residual_fs = generate_residual_frame(ref_gray, tar_gray, vectors_fs, block_size)
        residual_diamond = generate_residual_frame(ref_gray, tar_gray, vectors_diamond, block_size)

        # Compute energies
        e_fs = calculate_residual_energy(residual_fs)
        e_diamond = calculate_residual_energy(residual_diamond)

        energies_fs.append(e_fs)
        energies_diamond.append(e_diamond)

        # Save per-frame residual visualizations
        vis.save_residual_frame(residual_fs, "Full Search", frame_idx=idx)
        vis.save_residual_frame(residual_diamond, "Diamond Search", frame_idx=idx)
        vis.save_residual_comparison(residual_fs, residual_diamond, frame_idx=idx)

    energy_data = {
        'Full Search': energies_fs,
        'Diamond Search': energies_diamond,
    }

    # Save comparison chart
    chart_path = charts_dir / "residual_energy_series.png"
    plot_residual_energy_series(energy_data, chart_path)

    return energy_data
