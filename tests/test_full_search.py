"""Unit tests for Full Search motion estimation."""

from pathlib import Path
import tempfile

import numpy as np

from src.motion_estimation.full_search import full_search_motion_estimation
from src.motion_estimation.motion_vector import MotionVector


def test_full_search_returns_list() -> None:
    """Full Search should return a list of motion vectors."""
    reference = np.zeros((16, 16), dtype=np.uint8)
    target = np.zeros((16, 16), dtype=np.uint8)
    motion_vectors = full_search_motion_estimation(reference, target, block_size=8, search_range=2)
    assert isinstance(motion_vectors, list)
    assert all(isinstance(mv, MotionVector) for mv in motion_vectors)


def test_full_search_detects_block_displacement_and_saves_csv() -> None:
    """Full Search should detect a shifted block and optionally save motion vectors."""
    reference = np.zeros((16, 16), dtype=np.uint8)
    target = np.zeros((16, 16), dtype=np.uint8)
    reference[0:8, 0:8] = 255
    target[2:10, 2:10] = 255

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "motion_vectors.csv"
        motion_vectors = full_search_motion_estimation(
            reference,
            target,
            block_size=8,
            search_range=4,
            metric="mad",
            save_path=save_path,
        )

        assert len(motion_vectors) == 4
        first_vector = next((mv for mv in motion_vectors if mv.x == 0 and mv.y == 0), None)
        assert first_vector is not None
        assert first_vector.dx == 2
        assert first_vector.dy == 2
        assert save_path.exists()
        content = save_path.read_text(encoding="utf-8")
        assert "x,y,dx,dy" in content
