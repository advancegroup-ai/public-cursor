from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BBoxXYXY:
    """Pixel bbox in (x1, y1, x2, y2) inclusive-exclusive convention."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clip_to(self, width: int, height: int) -> "BBoxXYXY":
        x1 = max(0, min(self.x1, width))
        y1 = max(0, min(self.y1, height))
        x2 = max(0, min(self.x2, width))
        y2 = max(0, min(self.y2, height))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


ArrayF32 = np.ndarray
