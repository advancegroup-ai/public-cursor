from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .io import BBoxXYXY


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, eps, None)


class ImageEmbedder(Protocol):
    output_dim: int

    def embed(self, image_rgb: np.ndarray) -> np.ndarray: ...


@dataclass
class DeterministicImageEmbedder:
    """Simple deterministic baseline embedder for tests/dev."""

    output_dim: int = 512

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        arr = image_rgb.astype(np.float32)
        flat = arr.reshape(-1, 3)
        channel_means = flat.mean(axis=0)
        channel_stds = flat.std(axis=0)
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        weight = ((xx + 1.0) / (w + 1.0) + (yy + 1.0) / (h + 1.0)) / 2.0
        weight = weight[..., None]
        weighted_means = (arr * weight).reshape(-1, 3).mean(axis=0)
        vec = np.zeros((self.output_dim,), dtype=np.float32)
        feat = np.concatenate([channel_means, channel_stds, weighted_means], axis=0) / 255.0
        k = min(self.output_dim, feat.shape[0])
        vec[:k] = feat[:k]
        if self.output_dim > k:
            vec[k:] = feat.mean()
        return l2_normalize(vec)


@dataclass
class FaceMaskedDocFrontEmbedder:
    """Masks face region on doc_front then embeds the background."""

    base_embedder: ImageEmbedder

    def embed(self, image_rgb: np.ndarray, face_bbox: BBoxXYXY | None) -> np.ndarray:
        img = image_rgb.copy()
        if face_bbox is not None:
            h, w = img.shape[:2]
            box = face_bbox.clip(width=w, height=h)
            if box.is_valid():
                ys, xs = box.as_slices()
                img[ys, xs] = 0
        return self.base_embedder.embed(img)


@dataclass
class DeterministicFaceEmbedder:
    """ArcFace-like output contract (512-d normalized) for tests/dev."""

    output_dim: int = 512

    def get_feature(self, aligned_face_rgb_112: np.ndarray) -> np.ndarray:
        if aligned_face_rgb_112.shape[:2] != (112, 112):
            raise ValueError("face crop must be 112x112")
        arr = aligned_face_rgb_112.astype(np.float32)
        col_stat = arr.mean(axis=(0, 1)) / 255.0
        vec = np.zeros((self.output_dim,), dtype=np.float32)
        vec[:3] = col_stat
        if self.output_dim > 3:
            vec[3:] = col_stat.mean()
        return l2_normalize(vec)

