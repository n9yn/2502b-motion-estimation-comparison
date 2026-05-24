"""Unit tests for Full Search motion estimation."""

from src.motion_estimation.full_search import full_search_motion_estimation


def test_full_search_returns_list() -> None:
    """Full Search should return a list of motion vectors."""
    motion_vectors = full_search_motion_estimation(None, None)
    assert isinstance(motion_vectors, list)
