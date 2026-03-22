from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


class FaceBackend(Protocol):
    def embed_aligned_112(self, aligned_face_112: Image.Image) -> np.ndarray:  # pragma: no cover
        ...


@dataclass(frozen=True)
class ArcFaceEmbedder:
    backend: FaceBackend
    l2_normalize: bool = True

    def embed_aligned_112(self, aligned_face_112: Image.Image) -> np.ndarray:
        img = aligned_face_112.convert("RGB").resize((112, 112), resample=Image.BICUBIC)
        vec = np.asarray(self.backend.embed_aligned_112(img)).reshape(-1).astype(np.float32)
        if vec.shape[0] != 512:
            raise ValueError(f"Expected 512-d embedding, got shape {vec.shape}")
        if self.l2_normalize:
            vec = self._l2_normalize(vec)
        return vec

    @staticmethod
    def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        denom = float(np.linalg.norm(vec) + eps)
        return (vec / denom).astype(np.float32)

