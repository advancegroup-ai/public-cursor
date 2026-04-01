from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BBoxXYXY:
    """Pixel bbox in (x1, y1, x2, y2) format with x2/y2 exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, width: int, height: int) -> BBoxXYXY:
        x1 = int(np.clip(self.x1, 0, width))
        x2 = int(np.clip(self.x2, 0, width))
        y1 = int(np.clip(self.y1, 0, height))
        y2 = int(np.clip(self.y2, 0, height))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def mask_bbox_zero_rgb(img_rgb: np.ndarray, bbox: BBoxXYXY) -> np.ndarray:
    """
    Zero out pixels inside bbox.

    - img_rgb: uint8 RGB array shaped (H, W, 3)
    - bbox: (x1, y1, x2, y2) with x2/y2 exclusive
    """
    if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB; got {img_rgb.shape}")
    if img_rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB; got {img_rgb.dtype}")

    h, w, _ = img_rgb.shape
    b = bbox.clip(width=w, height=h)
    out = img_rgb.copy()
    if b.area() == 0:
        return out
    out[b.y1 : b.y2, b.x1 : b.x2, :] = 0
    return out

