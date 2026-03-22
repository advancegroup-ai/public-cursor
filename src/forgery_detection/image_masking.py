from __future__ import annotations

import numpy as np


def mask_bbox_zero_rgb(
    img: np.ndarray,
    bbox_xyxy: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """
    Zero-out pixels inside bbox on an RGB/BGR image.

    - img: HxWxC uint8/float image
    - bbox_xyxy: (x1, y1, x2, y2) in pixel coords (inclusive/exclusive tolerated)
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"img must be HxWxC with C>=3, got {img.shape}")
    b = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    if b.size != 4:
        raise ValueError(f"bbox must have 4 values, got {b.size}")

    h, w = img.shape[:2]
    x1, y1, x2, y2 = b.tolist()
    x1i = int(max(0, min(w, round(x1))))
    y1i = int(max(0, min(h, round(y1))))
    x2i = int(max(0, min(w, round(x2))))
    y2i = int(max(0, min(h, round(y2))))
    if x2i <= x1i or y2i <= y1i:
        return img

    out = img.copy()
    out[y1i:y2i, x1i:x2i, :3] = 0
    return out
