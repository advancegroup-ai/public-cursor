from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BBoxXYXY:
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, w: int, h: int) -> "BBoxXYXY":
        x1 = int(max(0, min(self.x1, w)))
        y1 = int(max(0, min(self.y1, h)))
        x2 = int(max(0, min(self.x2, w)))
        y2 = int(max(0, min(self.y2, h)))
        return BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)

    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1


def mask_bbox_zero_rgb(img_rgb_u8: np.ndarray, bbox: BBoxXYXY) -> np.ndarray:
    """
    Returns a copy of `img_rgb_u8` with the bbox region zeroed out.

    Expected input: uint8 RGB, shape (H, W, 3).
    """
    if img_rgb_u8.dtype != np.uint8:
        raise ValueError("img must be uint8")
    if img_rgb_u8.ndim != 3 or img_rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3) RGB, got {img_rgb_u8.shape}")

    h, w = img_rgb_u8.shape[0], img_rgb_u8.shape[1]
    b = bbox.clamp(w=w, h=h)
    out = img_rgb_u8.copy()
    if not b.is_valid():
        return out
    out[b.y1 : b.y2, b.x1 : b.x2, :] = 0
    return out

