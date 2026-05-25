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
    from typing import List
    import os
    import csv
    import time

    import numpy as np

    from src.motion_estimation.full_search import full_search_motion_estimation
    from src.motion_estimation.diamond_search import diamond_search_motion_estimation
    from src.residual.residual_generator import generate_residual
    from src.residual.residual_energy import residual_energy
    from src.visualization.comparison_chart import plot_residual_energy_series
    from src.visualization.visualization_manager import VisualizationManager
    from src.utils.metrics import mse, psnr


    def analyze_residuals(
        frames_dir: str,
        output_dir: str,
        block_size: int = 16,
        search_range: int = 8,
    ):
        os.makedirs(output_dir, exist_ok=True)
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        vm = VisualizationManager(output_dir)

        frame_files = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        energies = []
        rows = []

        for i in range(len(frame_files) - 1):
            ref_path = frame_files[i]
            tgt_path = frame_files[i + 1]
            ref = __import__("cv2").imread(ref_path)
            tgt = __import__("cv2").imread(tgt_path)

            fs_result = full_search_motion_estimation(ref, tgt, block_size=block_size, search_range=search_range, return_stats=True)
            if isinstance(fs_result, tuple):
                fs_vectors, fs_stats = fs_result
            else:
                fs_vectors = fs_result
                fs_stats = {"comparisons": 0, "time_ms": 0.0}

            ds_result = diamond_search_motion_estimation(ref, tgt, block_size=block_size, search_range=search_range, return_stats=True)
            if isinstance(ds_result, tuple):
                ds_vectors, ds_stats = ds_result
            else:
                ds_vectors = ds_result
                ds_stats = {"comparisons": 0, "time_ms": 0.0}

            # generate residuals using FS vectors
            predicted_fs = generate_residual(ref, tgt, fs_vectors, block_size=block_size, return_predicted=True)
            residual_fs = generate_residual(ref, tgt, fs_vectors, block_size=block_size, return_predicted=False)
            energy_fs = residual_energy(residual_fs)

            # compute accuracy metrics between target and predicted
            mse_fs = mse(tgt, predicted_fs)
            psnr_fs = psnr(tgt, predicted_fs)

            energies.append((i, energy_fs))

            # save visualizations
            vm.save_vector_field_image(fs_vectors, os.path.join(output_dir, f"fs_vectors_{i:03d}.png"))
            vm.save_vector_field_image(ds_vectors, os.path.join(output_dir, f"ds_vectors_{i:03d}.png"))
            vm.save_overlay_image(ref, fs_vectors, os.path.join(output_dir, f"fs_overlay_{i:03d}.png"))
            vm.save_overlay_image(ref, ds_vectors, os.path.join(output_dir, f"ds_overlay_{i:03d}.png"))
            vm.save_residual_image(residual_fs, os.path.join(output_dir, f"residual_fs_{i:03d}.png"))

            row = {
                "frame_index": i,
                "fs_comparisons": int(fs_stats.get("comparisons", 0)),
                "fs_time_ms": float(fs_stats.get("time_ms", 0.0)),
                "ds_comparisons": int(ds_stats.get("comparisons", 0)),
                "ds_time_ms": float(ds_stats.get("time_ms", 0.0)),
                "residual_energy_fs": float(energy_fs),
                "mse_fs": float(mse_fs),
                "psnr_fs": float(psnr_fs),
            }
            rows.append(row)

        # plot energy series
        if energies:
            indices, vals = zip(*energies)
            plot_residual_energy_series(indices, vals, os.path.join(output_dir, "residual_energy_series.png"))

        # write CSV report
        csv_path = os.path.join(reports_dir, "summary.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

        # write Markdown summary
        md_path = os.path.join(reports_dir, "summary.md")
        with open(md_path, "w") as md:
            md.write("# Residual Analysis Summary\n\n")
            md.write(f"Processed {len(rows)} frame pairs.\n\n")
            md.write("|frame_index|fs_comparisons|fs_time_ms|ds_comparisons|ds_time_ms|residual_energy_fs|mse_fs|psnr_fs|\n")
            md.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in rows:
                md.write(f"|{r['frame_index']}|{r['fs_comparisons']}|{r['fs_time_ms']:.2f}|{r['ds_comparisons']}|{r['ds_time_ms']:.2f}|{r['residual_energy_fs']:.2f}|{r['mse_fs']:.2f}|{r['psnr_fs']:.2f}|\n")

        print(f"Residual analysis completed. Outputs saved to {output_dir} and reports to {reports_dir}")

    # Save comparison chart
    chart_path = charts_dir / "residual_energy_series.png"
    plot_residual_energy_series(energy_data, chart_path)

    return energy_data
