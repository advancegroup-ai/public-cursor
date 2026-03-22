from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ArcFaceBackend(Protocol):
    """
    Backend interface for ArcFace-style face embedding.

    Input: aligned face tensor/image already prepared upstream as (112, 112, 3) RGB or
    whatever your backend expects; this abstraction keeps repo dependency-light.
    """

    def get_feature(self, aligned_face: np.ndarray) -> np.ndarray: ...


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(vec, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return vec / denom


@dataclass(frozen=True)
class ArcFaceEmbedder:
    backend: ArcFaceBackend
    l2_normalize: bool = True

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        vec = np.asarray(self.backend.get_feature(aligned_face)).astype(np.float32).reshape(-1)
        if vec.shape[0] != 512:
            raise ValueError(f"Expected 512-dim embedding, got shape {vec.shape}")
        if self.l2_normalize:
            vec = _l2_normalize(vec)
        return vec

