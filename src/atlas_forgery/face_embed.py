from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class FaceEmbedder(Protocol):
    dim: int

    def embed_aligned_112(self, bgr_u8_112: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DeterministicFaceEmbedder:
    """
    Deterministic fallback (NOT a real face model).
    Produces a 512-d vector derived from image bytes for testing/pipelines.
    """

    dim: int = 512

    def embed_aligned_112(self, bgr_u8_112: np.ndarray) -> np.ndarray:
        x = np.asarray(bgr_u8_112, dtype=np.uint8)
        if x.shape != (112, 112, 3):
            raise ValueError(f"Expected aligned BGR 112x112x3; got shape={x.shape}")
        # Simple byte-hash into 512 floats
        buf = x.reshape(-1).astype(np.uint32)
        acc = np.zeros(self.dim, dtype=np.float32)
        for i, v in enumerate(buf.tolist()):
            acc[i % self.dim] += float((v * 2654435761) & 0xFFFF) / 65535.0
        n = float(np.linalg.norm(acc)) or 1.0
        return (acc / n).astype(np.float32)


@dataclass(frozen=True)
class InsightFaceArcFaceEmbedder:
    """
    Optional ArcFace backend via insightface (if installed).
    Expects an aligned face crop (112,112,3) in BGR uint8.
    """

    dim: int = 512
    ctx_id: int = -1  # -1 CPU, >=0 GPU

    def _model(self):
        from insightface.model_zoo import model_zoo

        model = model_zoo.get_model("arcface_r100_v1", download=True)
        model.prepare(ctx_id=self.ctx_id)
        return model

    def embed_aligned_112(self, bgr_u8_112: np.ndarray) -> np.ndarray:
        x = np.asarray(bgr_u8_112, dtype=np.uint8)
        if x.shape != (112, 112, 3):
            raise ValueError(f"Expected aligned BGR 112x112x3; got shape={x.shape}")
        model = self._model()
        v = model.get_feat(x).reshape(-1).astype(np.float32)
        if v.size != self.dim:
            raise ValueError(f"Expected dim={self.dim}; got {v.size}")
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype(np.float32)

