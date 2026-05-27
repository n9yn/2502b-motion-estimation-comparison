"""Unit tests for experiment report generation."""

from pathlib import Path

from src.analysis.comparison import AlgorithmStats, ComparisonResult
from src.analysis.reporting import create_experiment_tables
from src.motion_estimation.motion_vector import MotionVector


def test_create_experiment_tables_outputs_files(tmp_path: Path) -> None:
    full_stats = AlgorithmStats(
        name="Full Search",
        execution_time_ms=120.5,
        comparisons=1024,
        residual_energy=1500.2,
        mse=12.34,
        psnr=28.56,
        num_vectors=16,
    )
    diamond_stats = AlgorithmStats(
        name="Diamond Search",
        execution_time_ms=42.1,
        comparisons=256,
        residual_energy=1520.8,
        mse=12.87,
        psnr=27.95,
        num_vectors=16,
    )
    comparison = ComparisonResult(full_search=full_stats, diamond_search=diamond_stats)

    results = create_experiment_tables(
        comparison,
        {
            "Full Search": [MotionVector(0, 0, 0, 0)],
            "Diamond Search": [MotionVector(0, 0, 1, 1)],
        },
        tmp_path,
    )

    assert (tmp_path / "runtime_comparison_table.csv").exists()
    assert (tmp_path / "runtime_comparison_table.md").exists()
    assert (tmp_path / "residual_energy_table.csv").exists()
    assert (tmp_path / "residual_energy_table.md").exists()
    assert (tmp_path / "motion_vector_comparison_table.csv").exists()
    assert (tmp_path / "motion_vector_comparison_table.md").exists()
    assert (tmp_path / "experiment_summary.md").exists()
    assert "runtime" in results
    assert "energy" in results
    assert "motion_vectors" in results
    assert "summary" in results
