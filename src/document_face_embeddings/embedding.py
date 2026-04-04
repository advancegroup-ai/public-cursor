from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .io import BBoxXYXY


class ImageEmbedder(Protocol):
    """Embeds an RGB uint8 image into a 1D float vector."""

    def embed(self, rgb: np.ndarray) -> np.ndarray:  # (D,)
        raise NotImplementedError


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= eps:
        return v * 0.0
    return v / n


@dataclass(frozen=True)
class MaskedEmbedder:
    """Applies zero-masking to a face region before embedding."""

    base: ImageEmbedder
    mask_value: int = 0  # 0 => black pixels

    def embed(self, rgb: np.ndarray, face_bbox: BBoxXYXY | None = None) -> np.ndarray:
        img = np.asarray(rgb, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB HxWx3, got shape={img.shape}")

        if face_bbox is not None:
            h, w = img.shape[:2]
            bb = face_bbox.clip(width=w, height=h)
            if not bb.is_empty():
                masked = img.copy()
                masked[bb.y1 : bb.y2, bb.x1 : bb.x2] = self.mask_value
                img = masked
        return self.base.embed(img)


@dataclass(frozen=True)
class MeanRGBEmbedder:
    """Tiny deterministic embedder for tests/baselines (3-dim mean RGB)."""

    normalize: bool = True

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(rgb, dtype=np.float32)
        vec = x.reshape(-1, 3).mean(axis=0)
        return l2_normalize(vec) if self.normalize else vec

