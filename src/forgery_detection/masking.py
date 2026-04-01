from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
 
 
@dataclass(frozen=True)
class BBox:
    """
    Pixel bbox in (x0, y0, x1, y1) with (x1,y1) exclusive.
    """
 
    x0: int
    y0: int
    x1: int
    y1: int
 
    def clipped(self, width: int, height: int) -> BBox:
        x0 = max(0, min(int(self.x0), width))
        x1 = max(0, min(int(self.x1), width))
        y0 = max(0, min(int(self.y0), height))
        y1 = max(0, min(int(self.y1), height))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)
 
 
def mask_bbox_zero_rgb(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """
    Zero out a bbox region in an HxWxC uint8/float image. Returns a copy.
    """
    if image.ndim != 3:
        raise ValueError(f"expected HxWxC image, got shape={image.shape}")
    h, w, _c = image.shape
    b = bbox.clipped(w, h)
    out = image.copy()
    if b.x1 > b.x0 and b.y1 > b.y0:
        out[b.y0 : b.y1, b.x0 : b.x1, :] = 0
    return out
