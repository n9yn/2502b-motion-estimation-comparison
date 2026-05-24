"""Unit tests for Diamond Search motion estimation."""

from src.motion_estimation.diamond_search import diamond_search_motion_estimation


def test_diamond_search_returns_list() -> None:
    """Diamond Search should return a list of motion vectors."""
    motion_vectors = diamond_search_motion_estimation(None, None)
    assert isinstance(motion_vectors, list)
