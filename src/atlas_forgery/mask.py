from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BBoxXYXY:
    """Integer XYXY box in pixel coordinates (x1, y1, x2, y2), x2/y2 exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clipped(self, width: int, height: int) -> BBoxXYXY:
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


def mask_bbox_zero_rgb(image_rgb: np.ndarray, bbox: BBoxXYXY) -> np.ndarray:
    """
    Return a copy of HxWx3 uint8/float image with bbox region zeroed.
    This mirrors the core intent of: img[y1:y2, x1:x2] = 0
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape={image_rgb.shape}")
    h, w, _ = image_rgb.shape
    box = bbox.clipped(width=w, height=h)
    out = image_rgb.copy()
    if not box.is_empty():
        out[box.y1 : box.y2, box.x1 : box.x2, :] = 0
    return out
