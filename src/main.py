"""Project entry point for motion estimation comparison."""

from pathlib import Path

from src.preprocessing.frame_extractor import (
    extract_frames_from_video,
    verify_frame_sequence_integrity,
)
from src.preprocessing.grayscale_converter import batch_convert_directory
from src.motion_estimation.full_search import full_search_motion_estimation
from src.motion_estimation.diamond_search import diamond_search_motion_estimation
from src.residual.residual_generator import generate_residual_frame
from src.residual.residual_energy import calculate_residual_energy
from src.visualization.vector_field import plot_motion_vectors
from src.visualization.comparison_chart import plot_runtime_comparison
from src.utils.config import Config
from src.utils.file_handler import resolve_working_paths


def main() -> None:
    """Run the motion estimation comparison workflow."""
    Config.validate()
    sample_video = Config.SAMPLE_VIDEO
    output_dir = resolve_working_paths()

    extracted_path = Config.EXTRACTED_FRAMES_DIR / sample_video.stem
    sample_output_path = output_dir["logs"] / "sample_frames"
    grayscale_output_path = Config.PROCESSED_DIR / f"{sample_video.stem}_grayscale"

    extract_frames_from_video(
        sample_video,
        extracted_path,
        sample_output_dir=sample_output_path,
        sample_frame_count=5,
    )

    sequence_ok = verify_frame_sequence_integrity(extracted_path)
    print(f"Frame sequence integrity: {'passed' if sequence_ok else 'failed'}")

    batch_convert_directory(extracted_path, grayscale_output_path)

    print("Motion estimation comparison workflow started.")
    print(f"Sample video: {sample_video}")
    print(f"Extracted frames directory: {extracted_path}")
    print(f"Saved sample frames to: {sample_output_path}")
    print(f"Saved grayscale frames to: {grayscale_output_path}")
    print("Review the outputs directory for generated artifacts.")


if __name__ == "__main__":
    main()
