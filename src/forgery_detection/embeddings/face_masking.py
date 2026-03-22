from __future__ import annotations

import numpy as np

from forgery_detection.types import BBoxXYXY


def zero_out_bbox(img_bgr: np.ndarray, bbox: BBoxXYXY) -> np.ndarray:
    """
    Returns a copy of img_bgr with bbox region set to 0.

    The upstream reference logic uses:
        img[y1:y2, x1:x2] = 0
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape={img_bgr.shape}")

    h, w = img_bgr.shape[:2]
    bb = bbox.clip_to(width=w, height=h)
    out = img_bgr.copy()
    if bb.area() == 0:
        return out
    out[bb.y1 : bb.y2, bb.x1 : bb.x2] = 0
    return out

