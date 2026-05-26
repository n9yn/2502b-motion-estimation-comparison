from pathlib import Path
import tempfile

from src.utils.config import Config
from src.workflow import run_motion_estimation_workflow


def test_low_motion_video_workflow_runs_and_outputs_exist() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        workflow_output = run_motion_estimation_workflow(
            Config.RAW_VIDEOS_DIR / "low_motion.mp4",
            output_root=tmp_dir,
        )

        assert Path(workflow_output["fs_vector_path"]).exists()
        assert Path(workflow_output["ds_vector_path"]).exists()
        assert Path(workflow_output["residual_image_path"]).exists()
        assert Path(workflow_output["residual_stats_path"]).exists()
        assert workflow_output["residual_stats"]["energy"] >= 0
        assert len(workflow_output["fs_vectors"]) > 0
        assert len(workflow_output["diamond_vectors"]) > 0


def test_high_motion_video_workflow_runs_and_outputs_exist() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        workflow_output = run_motion_estimation_workflow(
            Config.RAW_VIDEOS_DIR / "high_motion.mp4",
            output_root=tmp_dir,
        )

        assert Path(workflow_output["fs_vector_path"]).exists()
        assert Path(workflow_output["ds_vector_path"]).exists()
        assert Path(workflow_output["residual_image_path"]).exists()
        assert Path(workflow_output["residual_stats_path"]).exists()
        assert workflow_output["residual_stats"]["energy"] >= 0
        assert len(workflow_output["fs_vectors"]) > 0
        assert len(workflow_output["diamond_vectors"]) > 0
