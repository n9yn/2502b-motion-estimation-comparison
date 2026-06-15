from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.motion_estimation.diamond_search import diamond_search_motion_estimation
from src.motion_estimation.full_search import full_search_motion_estimation
from src.preprocessing.frame_extractor import extract_frames_from_video
from src.preprocessing.grayscale_converter import batch_convert_directory
from src.residual.residual_energy import save_residual_statistics
from src.residual.residual_generator import generate_residual_frame
from src.utils.config import Config
from src.utils.file_handler import resolve_working_paths
from src.visualization.comparison_chart import (
    plot_runtime_comparison,
    plot_energy_comparison,
)


def _resolve_working_paths(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    paths = {
        "root": root,
        "output_dir": root / "outputs",
        "motion_vectors": root / "outputs" / "motion_vectors",
        "residual_frames": root / "outputs" / "residual_frames",
        "vector_visualizations": root / "outputs" / "vector_visualizations",
        "charts": root / "outputs" / "charts",
        "logs": root / "outputs" / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _load_gray_frame(frame_path: Path) -> np.ndarray:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise FileNotFoundError(f"Unable to load frame: {frame_path}")
    return frame


def _get_sorted_frame_paths(directory: Path) -> list[Path]:
    return sorted([path for path in directory.glob("frame_*.png") if path.is_file()])


def _save_residual_image(residual_frame: np.ndarray, output_path: Path) -> None:
    residual = residual_frame.astype(np.float32)
    max_abs = float(np.max(np.abs(residual)))
    if max_abs == 0:
        image = np.full(residual.shape, 128, dtype=np.uint8)
    else:
        # Map signed residuals to 0-255 grayscale with zero at 128.
        image = np.clip((residual / max_abs) * 127.0 + 128.0, 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def run_motion_estimation_workflow(
    video_path: str | Path,
    block_size: int = Config.BLOCK_SIZE,
    search_range: int = Config.SEARCH_RANGE,
    use_numba: bool = False,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the motion estimation pipeline on a single video file."""
    Config.validate()
    paths = resolve_working_paths() if output_root is None else _resolve_working_paths(output_root)
    source_video = Path(video_path)
    if output_root is None:
        extracted_path = Config.EXTRACTED_FRAMES_DIR / source_video.stem
        grayscale_path = Config.PROCESSED_DIR / f"{source_video.stem}_grayscale"
    else:
        extracted_path = Path(output_root) / "data" / "extracted_frames" / source_video.stem
        extracted_path.mkdir(parents=True, exist_ok=True)
        grayscale_path = Path(output_root) / "data" / "processed" / f"{source_video.stem}_grayscale"
        grayscale_path.mkdir(parents=True, exist_ok=True)

    sample_path = paths["logs"] / f"{source_video.stem}_sample_frames"
    fs_vector_path = paths["motion_vectors"] / "fs" / f"{source_video.stem}_motion_vectors.csv"
    ds_vector_path = paths["motion_vectors"] / "diamond" / f"{source_video.stem}_motion_vectors.csv"
    residual_image_path = paths["residual_frames"] / f"{source_video.stem}_residual.png"
    residual_stats_path = paths["logs"] / f"{source_video.stem}_residual_stats.json"

    extract_frames_from_video(
        source_video,
        extracted_path,
        sample_output_dir=sample_path,
        sample_frame_count=5,
    )

    if not _get_sorted_frame_paths(extracted_path):
        raise RuntimeError(f"No extracted frames found for {source_video}")

    batch_convert_directory(extracted_path, grayscale_path)

    frame_paths = _get_sorted_frame_paths(extracted_path)
    if len(frame_paths) < 2:
        raise RuntimeError("At least two frames are required for motion estimation")

    reference_frame = _load_gray_frame(frame_paths[0])
    target_frame = _load_gray_frame(frame_paths[1])

    full_vectors, fs_stats = full_search_motion_estimation(
        reference_frame,
        target_frame,
        block_size=block_size,
        search_range=search_range,
        metric=Config.DEFAULT_METRIC,
        use_numba=use_numba,
        return_stats=True,
        save_path=fs_vector_path,
    )
    diamond_vectors, ds_stats = diamond_search_motion_estimation(
        reference_frame,
        target_frame,
        block_size=block_size,
        search_range=search_range,
        metric=Config.DEFAULT_METRIC,
        use_numba=use_numba,
        return_stats=True,
        save_path=ds_vector_path,
    )

    residual_frame_fs = generate_residual_frame(reference_frame, target_frame, full_vectors, block_size=block_size)
    residual_frame_ds = generate_residual_frame(reference_frame, target_frame, diamond_vectors, block_size=block_size)

    fs_residual_stats = save_residual_statistics(residual_frame_fs, residual_stats_path)
    ds_residual_stats_path = paths["logs"] / f"{source_video.stem}_diamond_residual_stats.json"
    ds_residual_stats = save_residual_statistics(residual_frame_ds, ds_residual_stats_path)

    fs_residual_image_path = paths["residual_frames"] / f"{source_video.stem}_fs_residual.png"
    ds_residual_image_path = paths["residual_frames"] / f"{source_video.stem}_diamond_residual.png"
    _save_residual_image(residual_frame_fs, fs_residual_image_path)
    _save_residual_image(residual_frame_ds, ds_residual_image_path)

    # Generate comparison charts
    runtime_chart_path = paths["charts"] / "runtime_comparison.png"
    energy_chart_path = paths["charts"] / "energy_comparison.png"
    
    runtime_data = {
        "Full Search": fs_stats.get("time_ms", 0),
        "Diamond Search": ds_stats.get("time_ms", 0),
    }
    plot_runtime_comparison(runtime_data, runtime_chart_path)
    
    energy_data = {
        "Full Search": fs_residual_stats.get("energy", 0),
        "Diamond Search": ds_residual_stats.get("energy", 0),
    }
    plot_energy_comparison(energy_data, energy_chart_path)

    return {
        "video_path": str(source_video),
        "extracted_path": str(extracted_path),
        "sample_path": str(sample_path),
        "grayscale_path": str(grayscale_path),
        "fs_vector_path": str(fs_vector_path),
        "ds_vector_path": str(ds_vector_path),
        "residual_image_path": str(fs_residual_image_path),
        "fs_residual_image_path": str(fs_residual_image_path),
        "ds_residual_image_path": str(ds_residual_image_path),
        "residual_stats_path": str(residual_stats_path),
        "ds_residual_stats_path": str(ds_residual_stats_path),
        "runtime_chart_path": str(runtime_chart_path),
        "energy_chart_path": str(energy_chart_path),
        "fs_vectors": full_vectors,
        "fs_stats": fs_stats,
        "diamond_vectors": diamond_vectors,
        "ds_stats": ds_stats,
        "residual_stats": fs_residual_stats,
        "diamond_residual_stats": ds_residual_stats,
    }
