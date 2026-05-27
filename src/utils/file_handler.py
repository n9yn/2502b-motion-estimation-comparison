"""Filesystem utilities for path resolution and file operations."""

from pathlib import Path


def resolve_working_paths() -> dict[str, Path]:
    """Create and return the common working paths for outputs and data."""
    project_root = Path(__file__).resolve().parents[2]
    paths = {
        "root": project_root,
        "output_dir": project_root / "outputs",
        "motion_vectors": project_root / "outputs" / "motion_vectors",
        "residual_frames": project_root / "outputs" / "residual_frames",
        "vector_visualizations": project_root / "outputs" / "vector_visualizations",
        "charts": project_root / "outputs" / "charts",
        "reports": project_root / "outputs" / "reports",
        "logs": project_root / "outputs" / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
