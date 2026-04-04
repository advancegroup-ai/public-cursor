from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, width: int, height: int) -> "BBox":
        x1 = int(np.clip(self.x1, 0, width))
        x2 = int(np.clip(self.x2, 0, width))
        y1 = int(np.clip(self.y1, 0, height))
        y2 = int(np.clip(self.y2, 0, height))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def is_empty(self) -> bool:
        return self.x2 <= self.x1 or self.y2 <= self.y1


def load_rgb_image(image: str | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def pil_to_rgb_numpy(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={arr.shape}")
    return arr


def zero_out_bbox(img_rgb: np.ndarray, bbox: Optional[BBox]) -> np.ndarray:
    """Returns a copy of img_rgb with bbox region set to 0 (black)."""
    if bbox is None:
        return img_rgb
    h, w = img_rgb.shape[:2]
    b = bbox.clamp(w, h)
    if b.is_empty():
        return img_rgb
    out = img_rgb.copy()
    out[b.y1 : b.y2, b.x1 : b.x2] = 0
    return out


def parse_bbox(text: str) -> BBox:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = (int(float(p)) for p in parts)
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

