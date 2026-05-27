"""Finalize the project by collecting outputs and verifying visual assets."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from src.utils.config import Config


def _collect_directory(source: Path, target: Path) -> list[Path]:
    copied_files: list[Path] = []
    if not source.exists():
        return copied_files

    for item in source.rglob("*"):
        if item.is_file():
            dest = target / item.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            copied_files.append(dest)
    return copied_files


def _verify_visuals(directories: list[Path]) -> dict[str, int]:
    verification = {}
    for directory in directories:
        if not directory.exists():
            verification[str(directory)] = 0
            continue
        count = sum(1 for _ in directory.rglob("*.png"))
        verification[str(directory)] = count
    return verification


def prepare_final_project(final_dir: str | Path | None = None) -> dict[str, Any]:
    """Prepare the final project folder with outputs, reports, and verified visuals."""
    Config.validate()

    if final_dir is None:
        final_dir = Config.FINAL_PROJECT_DIR
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    source_map = {
        "motion_vectors": Config.OUTPUT_DIR / "motion_vectors",
        "residual_frames": Config.OUTPUT_DIR / "residual_frames",
        "visualizations": Config.OUTPUT_DIR / "visualizations",
        "charts": Config.OUTPUT_DIR / "charts",
        "reports": Config.REPORTS_DIR,
        "demo_videos": Config.OUTPUT_DIR / "demo_videos",
        "demo_frames": Config.OUTPUT_DIR / "demo_frames",
        "demo_screenshots": Config.OUTPUT_DIR / "demo_screenshots",
    }

    copied: dict[str, int] = {}
    for name, source in source_map.items():
        destination = final_dir / name
        copied_files = _collect_directory(source, destination)
        copied[name] = len(copied_files)

    visuals_to_verify = [
        Config.OUTPUT_DIR / "visualizations",
        Config.OUTPUT_DIR / "charts",
        Config.OUTPUT_DIR / "demo_screenshots",
    ]
    verification = _verify_visuals(visuals_to_verify)

    report_path = final_dir / "final_project_summary.md"
    with open(report_path, "w", encoding="utf-8") as summary:
        summary.write("# Final Project Folder Summary\n\n")
        summary.write("## Copied Output Sections\n\n")
        for name, count in copied.items():
            summary.write(f"- {name}: {count} files\n")
        summary.write("\n## Visual Verification\n\n")
        for directory, count in verification.items():
            summary.write(f"- {directory}: {count} PNG files found\n")
        summary.write("\n## Final Folder\n\n")
        summary.write(f"Prepared final project folder: {final_dir}\n")

    return {
        "final_dir": final_dir,
        "copied": copied,
        "verification": verification,
        "summary_file": report_path,
    }


def main() -> None:
    print("Preparing final project folder...")
    result = prepare_final_project()
    print("\nFinal project preparation complete.")
    print(f"Final folder: {result['final_dir']}")
    print("Copied files:")
    for section, count in result["copied"].items():
        print(f"  {section}: {count}")
    print("\nVisual verification counts:")
    for directory, count in result["verification"].items():
        print(f"  {directory}: {count} PNG files")
    print(f"Summary file: {result['summary_file']}")


if __name__ == "__main__":
    main()
