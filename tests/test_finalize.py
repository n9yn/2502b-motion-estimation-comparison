"""Unit tests for final project packaging and verification."""

from pathlib import Path

from src.finalize import prepare_final_project
from src.utils.config import Config


def test_prepare_final_project_copies_existing_output_files(tmp_path: Path) -> None:
    original_output_dir = Config.OUTPUT_DIR
    original_reports_dir = Config.REPORTS_DIR
    original_final_dir = Config.FINAL_PROJECT_DIR

    try:
        Config.OUTPUT_DIR = tmp_path / "outputs"
        Config.REPORTS_DIR = Config.OUTPUT_DIR / "reports"
        Config.FINAL_PROJECT_DIR = tmp_path / "final_project"
        Config.validate()

        (Config.OUTPUT_DIR / "visualizations").mkdir(parents=True, exist_ok=True)
        (Config.OUTPUT_DIR / "charts").mkdir(parents=True, exist_ok=True)
        (Config.OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

        image_file = Config.OUTPUT_DIR / "visualizations" / "test.png"
        chart_file = Config.OUTPUT_DIR / "charts" / "chart.png"
        report_file = Config.OUTPUT_DIR / "reports" / "report.txt"

        image_file.write_bytes(b"dummy")
        chart_file.write_bytes(b"dummy")
        report_file.write_text("summary")

        result = prepare_final_project(final_dir=tmp_path / "final_project")

        assert (result["final_dir"] / "visualizations" / "test.png").exists()
        assert (result["final_dir"] / "charts" / "chart.png").exists()
        assert (result["final_dir"] / "reports" / "report.txt").exists()
        assert result["copied"]["visualizations"] == 1
        assert result["copied"]["charts"] == 1
        assert result["copied"]["reports"] == 1
        assert result["verification"][str(Config.OUTPUT_DIR / "visualizations")] == 1
        assert result["verification"][str(Config.OUTPUT_DIR / "charts")] == 1
        assert result["summary_file"].exists()
    finally:
        Config.OUTPUT_DIR = original_output_dir
        Config.REPORTS_DIR = original_reports_dir
        Config.FINAL_PROJECT_DIR = original_final_dir
