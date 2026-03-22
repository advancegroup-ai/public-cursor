from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBoxXYXY:
    """Pixel bbox in XYXY format: (x0, y0, x1, y1), with x1/y1 exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    def clipped(self, width: int, height: int) -> BBoxXYXY:
        x0 = max(0, min(self.x0, width))
        y0 = max(0, min(self.y0, height))
        x1 = max(0, min(self.x1, width))
        y1 = max(0, min(self.y1, height))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return BBoxXYXY(x0=x0, y0=y0, x1=x1, y1=y1)

    @property
    def is_empty(self) -> bool:
        return self.x1 <= self.x0 or self.y1 <= self.y0

