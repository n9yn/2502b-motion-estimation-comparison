"""Run the full demo workflow for clips, frames, and screenshots."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2

from src.demo.frame_selector import get_representative_frames
from src.demo.video_clips import create_all_demo_clips
from src.motion_estimation.diamond_search import diamond_search_motion_estimation
from src.motion_estimation.full_search import full_search_motion_estimation
from src.preprocessing.grayscale_converter import convert_to_grayscale
from src.residual.residual_generator import generate_residual_frame
from src.utils.config import Config
from src.visualization.visualization_manager import VisualizationManager


def save_selected_frames(
    selection_data: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """Copy selected representative frames to an output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    copied_frames: list[str] = []
    for frame_path in selection_data.get("selected_frames", []):
        source_path = Path(frame_path)
        if not source_path.exists():
            continue
        destination = output_path / source_path.name
        shutil.copy(source_path, destination)
        copied_frames.append(str(destination))

    return copied_frames


def create_demo_screenshots(
    video_path: str | Path,
    selection_data: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """Create screenshot visualizations for the selected representative frames."""
    video_path = Path(video_path)
    motion_level = video_path.stem
    frames_dir = Config.EXTRACTED_FRAMES_DIR / motion_level / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    output_path = Path(output_dir) / motion_level
    output_path.mkdir(parents=True, exist_ok=True)

    if len(frame_files) < 2:
        return []

    manager = VisualizationManager(output_path)
    screenshot_paths: list[str] = []

    for idx in selection_data.get("selected_indices", []):
        if idx < 0 or idx >= len(frame_files):
            continue

        frame_idx = min(idx, len(frame_files) - 2)
        frame_path = frame_files[frame_idx]
        next_frame_path = frame_files[frame_idx + 1]

        frame1_color = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        frame2_color = cv2.imread(str(next_frame_path), cv2.IMREAD_COLOR)
        if frame1_color is None or frame2_color is None:
            continue

        frame1 = convert_to_grayscale(frame1_color)
        frame2 = convert_to_grayscale(frame2_color)

        full_vectors, _ = full_search_motion_estimation(
            frame1,
            frame2,
            Config.BLOCK_SIZE,
            Config.SEARCH_RANGE,
            return_stats=True,
        )
        diamond_vectors, _ = diamond_search_motion_estimation(
            frame1,
            frame2,
            Config.BLOCK_SIZE,
            Config.SEARCH_RANGE,
            return_stats=True,
        )

        residual_fs = generate_residual_frame(frame1, frame2, full_vectors, Config.BLOCK_SIZE)
        residual_diamond = generate_residual_frame(frame1, frame2, diamond_vectors, Config.BLOCK_SIZE)

        saved = manager.save_all_visualizations(
            frame1_color,
            frame1_color,
            full_vectors,
            diamond_vectors,
            residual_fs,
            residual_diamond,
            frame_idx=frame_idx + 1,
        )

        screenshot_paths.extend(str(path) for path in saved.values())

    return screenshot_paths


def run_demo(
    clip_output_dir: str | Path | None = None,
    frame_output_dir: str | Path | None = None,
    screenshot_output_dir: str | Path | None = None,
    num_frames: int = 5,
    selection_method: str = "motion",
) -> dict[str, Any]:
    """Run the complete demo workflow for all videos in the raw data directory."""
    Config.validate()

    if clip_output_dir is None:
        clip_output_dir = Config.OUTPUT_DIR / "demo_videos"
    if frame_output_dir is None:
        frame_output_dir = Config.OUTPUT_DIR / "demo_frames"
    if screenshot_output_dir is None:
        screenshot_output_dir = Config.OUTPUT_DIR / "demo_screenshots"

    results: dict[str, Any] = {
        "clips": create_all_demo_clips(clip_output_dir),
        "representative_frames": [],
        "screenshots": [],
    }

    for video_file in sorted(Config.RAW_VIDEOS_DIR.glob("*.mp4")):
        selection_data = get_representative_frames(
            video_file,
            num_frames=num_frames,
            selection_method=selection_method,
        )

        frames_dir = Path(frame_output_dir) / video_file.stem
        copied_frames = save_selected_frames(selection_data, frames_dir)
        selection_data["copied_frames"] = copied_frames
        selection_data["output_dir"] = str(frames_dir)
        results["representative_frames"].append(selection_data)

        screenshots = create_demo_screenshots(video_file, selection_data, screenshot_output_dir)
        results["screenshots"].extend(screenshots)

    return results


def main() -> None:
    """Run the demo workflow from the command line."""
    print("\nRunning demo workflow...")
    results = run_demo()

    print("\nDemo workflow completed.")
    print(f"  clips created: {len(results['clips'].get('low_motion', [])) + len(results['clips'].get('medium_motion', [])) + len(results['clips'].get('high_motion', []))}")
    print(f"  comparison videos created: {len(results['clips'].get('comparisons', []))}")
    print(f"  representative frame sets: {len(results['representative_frames'])}")
    print(f"  screenshot files created: {len(results['screenshots'])}")
    print(f"\nCheck outputs at: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
