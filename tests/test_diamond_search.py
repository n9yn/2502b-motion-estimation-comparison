"""Unit tests for Diamond Search motion estimation."""

import numpy as np
from src.motion_estimation.diamond_search import diamond_search_motion_estimation


def test_diamond_search_returns_motion_vectors() -> None:
    """Diamond Search should return a list of motion vectors."""
    reference = np.zeros((32, 32), dtype=np.uint8)
    target = np.zeros((32, 32), dtype=np.uint8)
    target[0:16, 0:16] = 10

    motion_vectors = diamond_search_motion_estimation(reference, target, block_size=16, search_range=4)
    assert isinstance(motion_vectors, list)
    assert len(motion_vectors) == 4
    assert all(hasattr(mv, 'dx') and hasattr(mv, 'dy') for mv in motion_vectors)
