"""Utilities to extract residual energy series and save comparison charts and reports."""

import csv
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.motion_estimation.full_search import full_search_motion_estimation
from src.motion_estimation.diamond_search import diamond_search_motion_estimation
from src.residual.residual_generator import generate_residual_frame
from src.residual.residual_energy import calculate_residual_energy
from src.utils.metrics import compute_mse_from_residual, compute_psnr_from_residual
from src.visualization.comparison_chart import plot_residual_energy_series


def analyze_residuals(
    frames_dir: str | Path,
    output_dir: str | Path,
    block_size: int = 16,
    search_range: int = 8,
) -> dict[str, str]:
    """Analyze residual energy across consecutive frame pairs and save reports."""
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(
        [
            frames_dir / f
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    rows = []
    energy_data = {"Full Search": [], "Diamond Search": []}
    indices = []

    for i in range(len(frame_files) - 1):
        ref_path = frame_files[i]
        tgt_path = frame_files[i + 1]
        ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        tgt = cv2.imread(str(tgt_path), cv2.IMREAD_COLOR)

        if ref is None or tgt is None:
            continue

        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)

        fs_vectors, fs_stats = full_search_motion_estimation(
            ref_gray,
            tgt_gray,
            block_size=block_size,
            search_range=search_range,
            return_stats=True,
        )

        ds_vectors, ds_stats = diamond_search_motion_estimation(
            ref_gray,
            tgt_gray,
            block_size=block_size,
            search_range=search_range,
            return_stats=True,
        )

        residual_fs = generate_residual_frame(ref_gray, tgt_gray, fs_vectors, block_size)
        residual_diamond = generate_residual_frame(ref_gray, tgt_gray, ds_vectors, block_size)

        energy_fs = calculate_residual_energy(residual_fs)
        energy_ds = calculate_residual_energy(residual_diamond)
        mse_fs = compute_mse_from_residual(residual_fs)
        psnr_fs = compute_psnr_from_residual(residual_fs)

        rows.append(
            {
                "frame_index": i,
                "fs_comparisons": int(fs_stats.get("comparisons", 0)),
                "fs_time_ms": float(fs_stats.get("time_ms", 0.0)),
                "ds_comparisons": int(ds_stats.get("comparisons", 0)),
                "ds_time_ms": float(ds_stats.get("time_ms", 0.0)),
                "residual_energy_fs": float(energy_fs),
                "residual_energy_ds": float(energy_ds),
                "mse_fs": float(mse_fs),
                "psnr_fs": float(psnr_fs),
            }
        )
        energy_data["Full Search"].append(energy_fs)
        energy_data["Diamond Search"].append(energy_ds)
        indices.append(i)

    chart_path = output_dir / "residual_energy_series.png"
    if energy_data["Full Search"] or energy_data["Diamond Search"]:
        plot_residual_energy_series(energy_data, chart_path)

    csv_path = reports_dir / "residual_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    md_path = reports_dir / "residual_summary.md"
    with open(md_path, "w", encoding="utf-8") as md:
        md.write("# Residual Analysis Summary\n\n")
        md.write(f"Processed {len(rows)} frame pairs.\n\n")
        md.write("|frame_index|fs_comparisons|fs_time_ms|ds_comparisons|ds_time_ms|residual_energy_fs|residual_energy_ds|mse_fs|psnr_fs|\n")
        md.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            md.write(
                f"|{r['frame_index']}|{r['fs_comparisons']}|{r['fs_time_ms']:.2f}|{r['ds_comparisons']}|{r['ds_time_ms']:.2f}|"
                f"{r['residual_energy_fs']:.2f}|{r['residual_energy_ds']:.2f}|{r['mse_fs']:.2f}|{r['psnr_fs']:.2f}|\n"
            )

    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "chart": str(chart_path),
    }
