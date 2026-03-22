from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .similarity import l2_normalize


class FaceEmbedder(Protocol):
    def embed_aligned_112(self, face_rgb_u8: np.ndarray) -> np.ndarray:
        """Input: aligned face crop 112x112 RGB uint8. Output: float32 (512,)."""


@dataclass(frozen=True)
class DeterministicFaceEmbedder:
    out_dim: int = 512

    def embed_aligned_112(self, face_rgb_u8: np.ndarray) -> np.ndarray:
        if face_rgb_u8.shape[:2] != (112, 112):
            raise ValueError(f"Expected 112x112, got {face_rgb_u8.shape}")
        x = face_rgb_u8.astype(np.float32) / 255.0
        mean = x.mean(axis=(0, 1))  # 3
        v = np.zeros((self.out_dim,), dtype=np.float32)
        v[:3] = mean
        return l2_normalize(v)


@dataclass
class InsightFaceArcFaceEmbedder:
    """
    ArcFace embedding via insightface.

    This expects pre-aligned 112x112 faces. Detection/alignment is intentionally out of scope here.
    """

    provider: str = "CPUExecutionProvider"

    def __post_init__(self) -> None:
        try:
            from insightface.model_zoo import get_model  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("insightface is required for InsightFaceArcFaceEmbedder") from e

        # Using a common ArcFace model name; callers can pin/replace as needed.
        self._model = get_model("arcface_r100_v1", providers=[self.provider])
        self._model.prepare(ctx_id=-1)

    def embed_aligned_112(self, face_rgb_u8: np.ndarray) -> np.ndarray:
        if face_rgb_u8.shape[:2] != (112, 112):
            raise ValueError(f"Expected 112x112, got {face_rgb_u8.shape}")
        # insightface expects BGR by default for many models; convert RGB->BGR
        bgr = face_rgb_u8[:, :, ::-1]
        feat = self._model.get_feat(bgr).reshape(-1).astype(np.float32, copy=False)
        return l2_normalize(feat)

