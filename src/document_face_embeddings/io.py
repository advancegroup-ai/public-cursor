from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image as uint8 RGB array with shape (H, W, 3)."""
    p = Path(path)
    with Image.open(p) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got array shape={arr.shape} for {p}")
    return arr


@dataclass(frozen=True)
class BBoxXYXY:
    """Axis-aligned bounding box in pixel coords (x1,y1,x2,y2), half-open on the max edge."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, width: int, height: int) -> "BBoxXYXY":
        x1 = int(np.clip(self.x1, 0, width))
        x2 = int(np.clip(self.x2, 0, width))
        y1 = int(np.clip(self.y1, 0, height))
        y2 = int(np.clip(self.y2, 0, height))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)

    def is_empty(self) -> bool:
        return self.x2 <= self.x1 or self.y2 <= self.y1

