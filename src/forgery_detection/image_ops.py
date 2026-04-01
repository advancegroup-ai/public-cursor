from __future__ import annotations

import numpy as np


def mask_bbox_zero_rgb(
    img: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int]
) -> np.ndarray:
    """
    Zero-out a face bounding box from RGB/BGR image.

    Equivalent to:
    img[y0:y1, x0:x1] = 0
    """
    if img.ndim != 3:
        raise ValueError(f"expected HxWxC image, got shape {img.shape}")

    h, w = img.shape[:2]
    x0, y0 = top_left
    x1, y1 = bottom_right

    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))

    out = img.copy()
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 0
    return out
