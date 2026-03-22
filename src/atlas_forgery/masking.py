from __future__ import annotations

import numpy as np

from .types import BBoxXYXY


def mask_bbox_zero_rgb(img: np.ndarray, bbox: BBoxXYXY) -> np.ndarray:
    """
    Return a copy of img with bbox region zeroed.

    Expects HxWx3 uint8/float arrays. bbox is clipped to image bounds.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={img.shape}")

    h, w, _ = img.shape
    bb = bbox.clipped(width=w, height=h)
    out = img.copy()
    if bb.is_empty:
        return out
    out[bb.y0 : bb.y1, bb.x0 : bb.x1, :] = 0
    return out

