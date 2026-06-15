"""Motion vector representation for block-based motion estimation."""

from dataclasses import dataclass


@dataclass
class MotionVector:
    x: int
    y: int
    dx: int
    dy: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return a tuple representation of the motion vector."""
        return self.x, self.y, self.dx, self.dy

    def __str__(self) -> str:
        return f"MV(x={self.x}, y={self.y}, dx={self.dx}, dy={self.dy})"
