"""Configuration constants and project paths."""

from pathlib import Path


class Config:
    """Shared configuration values for the project."""

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
    EXTRACTED_FRAMES_DIR = DATA_DIR / "extracted_frames"
    PROCESSED_DIR = DATA_DIR / "processed"

    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    MOTION_VECTORS_DIR = OUTPUT_DIR / "motion_vectors"
    RESIDUAL_FRAMES_DIR = OUTPUT_DIR / "residual_frames"
    VECTOR_VISUALIZATIONS_DIR = OUTPUT_DIR / "vector_visualizations"
    CHARTS_DIR = OUTPUT_DIR / "charts"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    FINAL_PROJECT_DIR = OUTPUT_DIR / "final_project"
    LOGS_DIR = OUTPUT_DIR / "logs"

    SAMPLE_VIDEO = RAW_VIDEOS_DIR / "low_motion.mp4"
    BLOCK_SIZE = 16
    SEARCH_RANGE = 8
    DEFAULT_METRIC = "sad"

    @classmethod
    def validate(cls) -> None:
        """Validate that key directories exist or can be created."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        cls.EXTRACTED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.FINAL_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
