"""Full Search motion estimation algorithm placeholder."""

from pathlib import Path
from typing import Any

from src.motion_estimation.motion_vector import MotionVector


def full_search_motion_estimation(
    reference_frame: Any,
    target_frame: Any,
    block_size: int = 16,
    search_range: int = 8,
) -> list[MotionVector]:
    """Compute motion vectors using a full search algorithm."""
    # TODO: implement exhaustive full search block matching
    print("[TODO] Running Full Search motion estimation")
    return []
