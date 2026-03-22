from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class FaceEmbeddingBackend(Protocol):
    """
    A backend that maps aligned 112x112 RGB faces to a 512-d embedding.
    """

    def get_feature(self, aligned_face_rgb: np.ndarray) -> np.ndarray: ...


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(vec) + eps
    return vec / denom


@dataclass(frozen=True)
class ArcFaceEmbedder:
    backend: FaceEmbeddingBackend
    l2_normalize: bool = True

    def embed_aligned(self, aligned_face_rgb: np.ndarray) -> np.ndarray:
        vec = np.asarray(self.backend.get_feature(aligned_face_rgb)).reshape(-1).astype(np.float32)
        if vec.shape[0] != 512:
            raise ValueError(f"Expected 512-d embedding, got shape {vec.shape}")
        if self.l2_normalize:
            vec = _l2_normalize(vec)
        return vec

