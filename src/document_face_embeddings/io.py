from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BBoxXYXY:
    """Axis-aligned bounding box in (x1, y1, x2, y2) pixel coords.

    Convention: (x1, y1) inclusive, (x2, y2) exclusive. All coords integer.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def clipped(self, width: int, height: int) -> "BBoxXYXY":
        x1 = max(0, min(self.x1, width))
        x2 = max(0, min(self.x2, width))
        y1 = max(0, min(self.y1, height))
        y2 = max(0, min(self.y2, height))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def load_rgb(path: str | Path) -> np.ndarray:
    p = Path(path)
    img = Image.open(p).convert("RGB")
    return np.asarray(img)

