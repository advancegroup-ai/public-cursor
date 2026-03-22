from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


@dataclass(frozen=True)
class BBoxXYXY:
    """Pixel-space bounding box: (x1, y1, x2, y2), x2/y2 exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, width: int, height: int) -> "BBoxXYXY":
        x1 = int(max(0, min(self.x1, width)))
        y1 = int(max(0, min(self.y1, height)))
        x2 = int(max(0, min(self.x2, width)))
        y2 = int(max(0, min(self.y2, height)))
        return BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)

    def as_slices(self) -> Tuple[slice, slice]:
        return slice(self.y1, self.y2), slice(self.x1, self.x2)

    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

