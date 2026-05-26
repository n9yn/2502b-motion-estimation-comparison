"""Project entry point for motion estimation comparison."""

from src.utils.config import Config
from src.workflow import run_motion_estimation_workflow


def main() -> None:
    """Run the motion estimation comparison workflow."""
    Config.validate()
    sample_video = Config.SAMPLE_VIDEO
    results = run_motion_estimation_workflow(sample_video)

    print("Motion estimation comparison workflow completed.")
    print(f"Sample video: {results['video_path']}")
    print(f"Extracted frames directory: {results['extracted_path']}")
    print(f"Sample frames directory: {results['sample_path']}")
    print(f"Grayscale frames directory: {results['grayscale_path']}")
    print(f"Full Search motion vectors: {results['fs_vector_path']}")
    print(f"Diamond Search motion vectors: {results['ds_vector_path']}")
    print(f"Residual image: {results['residual_image_path']}")
    print(f"Residual statistics: {results['residual_stats_path']}")
    print(f"Residual energy: {results['residual_stats']['energy']}")


if __name__ == "__main__":
    main()
