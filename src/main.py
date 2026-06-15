"""Project entry point for motion estimation comparison."""

from pathlib import Path

import cv2

from src.preprocessing.frame_extractor import extract_frames_from_video
from src.preprocessing.grayscale_converter import convert_to_grayscale
from src.motion_estimation.full_search import full_search_motion_estimation
from src.motion_estimation.diamond_search import diamond_search_motion_estimation
from src.residual.residual_generator import generate_residual_frame
from src.residual.residual_energy import calculate_residual_energy
from src.visualization.visualization_manager import VisualizationManager
from src.analysis.comparison import create_comparison_result
from src.analysis.reporting import create_experiment_tables
from src.visualization.comparison_chart import plot_runtime_comparison, plot_energy_comparison
from src.utils.config import Config
from src.utils.file_handler import resolve_working_paths


def main() -> None:
    """Run the motion estimation comparison workflow."""
    Config.validate()
    sample_video = Config.SAMPLE_VIDEO
    resolve_working_paths()

    print("=" * 60)
    print("Motion Estimation Comparison Workflow")
    print("=" * 60)

    # Extract frames from video
    print("\n[1/6] Extracting frames from video...")
    extract_frames_from_video(sample_video, Config.EXTRACTED_FRAMES_DIR)
    print(f"✓ Frames extracted to {Config.EXTRACTED_FRAMES_DIR}")

    # Load sample frames for analysis
    frames_dir = Path(Config.EXTRACTED_FRAMES_DIR)
    frame_files = sorted(frames_dir.glob("*.png"))[:2]

    if len(frame_files) < 2:
        print("⚠ Warning: Could not find enough frames for comparison")
        return

    # Load frames
    print("\n[2/6] Loading frames for analysis...")
    frame1_color = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
    frame2_color = cv2.imread(str(frame_files[1]), cv2.IMREAD_COLOR)

    if frame1_color is None or frame2_color is None:
        print("✗ Error: Could not load frames")
        return

    frame1 = convert_to_grayscale(frame1_color)
    frame2 = convert_to_grayscale(frame2_color)
    print(f"✓ Loaded frames with shape {frame1.shape}")

    # Run motion estimation algorithms with statistics
    print("\n[3/6] Running motion estimation algorithms...")
    print("  Running Full Search...")
    full_vectors, fs_stats = full_search_motion_estimation(
        frame1,
        frame2,
        Config.BLOCK_SIZE,
        Config.SEARCH_RANGE,
        metric=Config.DEFAULT_METRIC,
        return_stats=True,
    )
    
    print("  Running Diamond Search...")
    diamond_vectors, diamond_stats = diamond_search_motion_estimation(
        frame1,
        frame2,
        Config.BLOCK_SIZE,
        Config.SEARCH_RANGE,
        metric=Config.DEFAULT_METRIC,
        return_stats=True,
    )
    
    print(f"✓ Full Search: {len(full_vectors)} motion vectors")
    print(f"✓ Diamond Search: {len(diamond_vectors)} motion vectors")

    # Generate residual frames
    print("\n[4/6] Generating residuals and calculating energy...")
    residual_fs = generate_residual_frame(frame1, frame2, full_vectors, Config.BLOCK_SIZE)
    residual_diamond = generate_residual_frame(frame1, frame2, diamond_vectors, Config.BLOCK_SIZE)

    energy_fs = calculate_residual_energy(residual_fs)
    energy_diamond = calculate_residual_energy(residual_diamond)

    print(f"✓ Full Search residual energy: {energy_fs:.2f}")
    print(f"✓ Diamond Search residual energy: {energy_diamond:.2f}")

    # Create comprehensive comparison results
    print("\n[5/6] Analyzing and comparing results...")
    comparison = create_comparison_result(
        frame1,
        frame2,
        residual_fs,
        residual_diamond,
        energy_fs,
        energy_diamond,
        (full_vectors, fs_stats),
        (diamond_vectors, diamond_stats),
    )
    comparison.print_summary()

    # Save all visualizations
    print("\n[6/6] Generating and saving visualizations...")
    vis_manager = VisualizationManager(Config.OUTPUT_DIR / "visualizations")

    results = vis_manager.save_all_visualizations(
        frame1_color,
        frame2_color,
        full_vectors,
        diamond_vectors,
        residual_fs,
        residual_diamond,
        frame_idx=0,
    )

    print(f"\n✓ Saved {len(results)} visualization files")

    # Save comparison charts with actual measured data
    print("\nGenerating comparison charts...")
    runtime_data = {
        'Full Search': fs_stats["time_ms"],
        'Diamond Search': diamond_stats["time_ms"],
    }
    energy_data = {
        'Full Search': energy_fs,
        'Diamond Search': energy_diamond,
    }

    vis_manager.save_runtime_comparison(runtime_data)
    vis_manager.save_energy_comparison(energy_data)

    # Generate experiment tables and organized reports
    print("\nOrganizing experimental result tables...")
    table_outputs = create_experiment_tables(
        comparison,
        {
            "Full Search": full_vectors,
            "Diamond Search": diamond_vectors,
        },
        Config.REPORTS_DIR,
    )
    print(f"✓ Saved experiment tables and summary to {Config.REPORTS_DIR}")

    # Print summary
    print("\n" + "=" * 60)
    print("Visualization Summary")
    print("=" * 60)
    summary = vis_manager.get_summary()
    for category, count in summary.items():
        print(f"  {category.capitalize():20} : {count:3} files")

    print("\nOutput directory:", Config.OUTPUT_DIR / "visualizations")
    print("\n✓ Motion estimation comparison workflow completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
