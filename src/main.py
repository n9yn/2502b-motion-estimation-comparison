"""Project entry point for motion estimation comparison."""

from pathlib import Path

from src.preprocessing.frame_extractor import extract_frames_from_video
from src.preprocessing.grayscale_converter import convert_to_grayscale
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

    # TODO: Load a sample video and run preprocessing
    extract_frames_from_video(sample_video, Config.EXTRACTED_FRAMES_DIR)

    # TODO: Run Full Search and Diamond Search motion estimation
    # Use the first two extracted frames for initial comparison
    # full_vectors = full_search_motion_estimation(...)
    # diamond_vectors = diamond_search_motion_estimation(...)

    # TODO: Compute residual frames and energy metrics
    # residual_frame = generate_residual_frame(...)
    # energy_value = calculate_residual_energy(residual_frame)

    # TODO: Generate visualizations and comparison charts
    # plot_motion_vectors(...)
    # plot_runtime_comparison(...)

    print("Motion estimation comparison workflow started.")
    print(f"Sample video: {sample_video}")
    print(f"Extracted frames directory: {Config.EXTRACTED_FRAMES_DIR}")
    print("Review the outputs directory for generated artifacts.")


if __name__ == "__main__":
    main()
