"""Unit tests for Diamond Search motion estimation."""

from pathlib import Path
import tempfile

import numpy as np

from src.motion_estimation.diamond_search import diamond_search_motion_estimation
from src.motion_estimation.full_search import full_search_motion_estimation
from src.motion_estimation.motion_vector import MotionVector


def test_diamond_search_returns_list() -> None:
    """Diamond Search should return a list of motion vectors."""
    reference = np.zeros((16, 16), dtype=np.uint8)
    target = np.zeros((16, 16), dtype=np.uint8)
    motion_vectors = diamond_search_motion_estimation(reference, target, block_size=8, search_range=2)
    assert isinstance(motion_vectors, list)
    assert all(isinstance(mv, MotionVector) for mv in motion_vectors)


def test_diamond_search_finds_shifted_block_and_matches_full_search() -> None:
    """Diamond Search should detect block displacement and compare with Full Search."""
    reference = np.zeros((16, 16), dtype=np.uint8)
    target = np.zeros((16, 16), dtype=np.uint8)
    reference[0:8, 0:8] = 255
    target[2:10, 2:10] = 255

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "diamond_motion_vectors.csv"
        diamond_vectors = diamond_search_motion_estimation(
            reference,
            target,
            block_size=8,
            search_range=4,
            metric="mad",
            save_path=save_path,
        )

        full_vectors = full_search_motion_estimation(
            reference,
            target,
            block_size=8,
            search_range=4,
            metric="mad",
        )

        assert len(diamond_vectors) == len(full_vectors)
        assert save_path.exists()

        first_diamond = next((mv for mv in diamond_vectors if mv.x == 0 and mv.y == 0), None)
        first_full = next((mv for mv in full_vectors if mv.x == 0 and mv.y == 0), None)
        assert first_diamond is not None
        assert first_full is not None
        assert first_diamond.dx == first_full.dx == 2
        assert first_diamond.dy == first_full.dy == 2
