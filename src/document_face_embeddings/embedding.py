from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import BBoxXYXY


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        denom = float(np.linalg.norm(x) + eps)
        return x / denom
    denom = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / denom


def mask_face_region(img_rgb: np.ndarray, face_bbox: BBoxXYXY | None) -> np.ndarray:
    img = np.asarray(img_rgb).copy()
    if face_bbox is None:
        return img
    h, w = img.shape[0], img.shape[1]
    bb = face_bbox.clipped(width=w, height=h)
    if bb.area() == 0:
        return img
    img[bb.y1 : bb.y2, bb.x1 : bb.x2] = 0
    return img


class Embedder:
    dim: int

    def embed(self, img_rgb: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class MeanRGBEmbedder(Embedder):
    """Deterministic baseline "embedding" useful for plumbing/tests.

    Produces a 3-dim vector = mean RGB / 255, L2-normalized.
    """

    dim: int = 3

    def embed(self, img_rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(img_rgb, dtype=np.float32)
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError("Expected HxWx3 RGB image")
        v = x.mean(axis=(0, 1)) / 255.0
        return l2_normalize(v)


@dataclass(frozen=True)
class MaskedEmbedder:
    base: Embedder

    @property
    def dim(self) -> int:
        return int(self.base.dim)

    def embed(self, img_rgb: np.ndarray, face_bbox: BBoxXYXY | None) -> np.ndarray:
        masked = mask_face_region(img_rgb, face_bbox=face_bbox)
        return self.base.embed(masked)

