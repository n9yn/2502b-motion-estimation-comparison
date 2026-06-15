"""Generate tabular experiment reports for the motion estimation comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.comparison import ComparisonResult, AlgorithmStats
from src.motion_estimation.motion_vector import MotionVector


def _compute_vector_metrics(vectors: list[MotionVector]) -> dict[str, float]:
    """Compute summary statistics for a motion vector list."""
    if len(vectors) == 0:
        return {
            "mean_magnitude": 0.0,
            "std_magnitude": 0.0,
            "max_magnitude": 0.0,
            "min_magnitude": 0.0,
            "zero_vectors": 0,
            "zero_ratio": 0.0,
        }

    magnitudes = np.array([np.hypot(mv.dx, mv.dy) for mv in vectors], dtype=np.float64)
    zero_vectors = int(np.sum(magnitudes == 0))
    zero_ratio = zero_vectors / len(vectors)

    return {
        "mean_magnitude": float(np.mean(magnitudes)),
        "std_magnitude": float(np.std(magnitudes)),
        "max_magnitude": float(np.max(magnitudes)),
        "min_magnitude": float(np.min(magnitudes)),
        "zero_vectors": zero_vectors,
        "zero_ratio": float(zero_ratio),
    }


def _save_table(df: pd.DataFrame, base_name: str, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{base_name}.csv"
    md_path = output_dir / f"{base_name}.md"
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write(df.to_markdown(index=False))
    return {"csv": csv_path, "md": md_path}


def create_experiment_tables(
    comparison: ComparisonResult,
    motion_vectors: dict[str, list[MotionVector]],
    output_dir: str | Path,
) -> dict[str, dict[str, Path]]:
    """Create runtime, energy, and motion vector comparison tables."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    algorithms = [comparison.full_search, comparison.diamond_search]
    rows: list[dict[str, Any]] = []

    for algorithm in algorithms:
        vector_metrics = _compute_vector_metrics(motion_vectors.get(algorithm.name, []))
        rows.append(
            {
                "Algorithm": algorithm.name,
                "Execution Time (ms)": algorithm.execution_time_ms,
                "Comparisons": algorithm.comparisons,
                "Residual Energy": algorithm.residual_energy,
                "MSE": algorithm.mse,
                "PSNR (dB)": algorithm.psnr,
                "Motion Vector Count": algorithm.num_vectors,
                "Mean Vector Magnitude": vector_metrics["mean_magnitude"],
                "Std Vector Magnitude": vector_metrics["std_magnitude"],
                "Max Vector Magnitude": vector_metrics["max_magnitude"],
                "Min Vector Magnitude": vector_metrics["min_magnitude"],
                "Zero Vector Count": vector_metrics["zero_vectors"],
                "Zero Vector Ratio": vector_metrics["zero_ratio"],
            }
        )

    df = pd.DataFrame(rows)

    runtime_df = df[["Algorithm", "Execution Time (ms)", "Comparisons"]]
    energy_df = df[["Algorithm", "Residual Energy", "MSE", "PSNR (dB)"]]
    vector_df = df[
        [
            "Algorithm",
            "Motion Vector Count",
            "Mean Vector Magnitude",
            "Std Vector Magnitude",
            "Max Vector Magnitude",
            "Min Vector Magnitude",
            "Zero Vector Count",
            "Zero Vector Ratio",
        ]
    ]

    results = {
        "runtime": _save_table(runtime_df, "runtime_comparison_table", output_path),
        "energy": _save_table(energy_df, "residual_energy_table", output_path),
        "motion_vectors": _save_table(vector_df, "motion_vector_comparison_table", output_path),
    }

    summary_path = output_path / "experiment_summary.md"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("# Experiment Comparison Summary\n\n")
        summary_file.write("## Overview\n\n")
        summary_file.write(
            f"- Full Search runtime: {comparison.full_search.execution_time_ms:.2f} ms\n"
        )
        summary_file.write(
            f"- Diamond Search runtime: {comparison.diamond_search.execution_time_ms:.2f} ms\n"
        )
        summary_file.write(
            f"- Full Search residual energy: {comparison.full_search.residual_energy:.2f}\n"
        )
        summary_file.write(
            f"- Diamond Search residual energy: {comparison.diamond_search.residual_energy:.2f}\n"
        )
        summary_file.write(
            f"- Full Search motion vectors: {comparison.full_search.num_vectors}\n"
        )
        summary_file.write(
            f"- Diamond Search motion vectors: {comparison.diamond_search.num_vectors}\n\n"
        )

        summary_file.write("## Generated Tables\n\n")
        summary_file.write("- `runtime_comparison_table.csv` / `.md`\n")
        summary_file.write("- `residual_energy_table.csv` / `.md`\n")
        summary_file.write("- `motion_vector_comparison_table.csv` / `.md`\n\n")

        summary_file.write("## Observations\n\n")
        summary_file.write(
            f"- Diamond Search speedup: {comparison.get_speedup():.2f}x\n"
        )
        summary_file.write(
            f"- Comparison reduction: {comparison.get_complexity_reduction():.1f}% fewer comparisons\n"
        )
        summary_file.write(
            f"- Residual energy ratio (DS / FS): {comparison.get_energy_ratio():.3f}\n"
        )
        summary_file.write(
            f"- MSE difference: {comparison.get_accuracy_loss():.2f}\n"
        )
        summary_file.write("\n## Table Links\n\n")
        summary_file.write(
            f"- `runtime_comparison_table.csv`\n"
            f"- `residual_energy_table.csv`\n"
            f"- `motion_vector_comparison_table.csv`\n"
        )

    results["summary"] = {"md": summary_path}
    return results
