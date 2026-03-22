from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class FaceEmbedder(Protocol):
    def embed_aligned(self, face_rgb_112: np.ndarray) -> np.ndarray:  # (512,)
        ...


@dataclass(frozen=True)
class MeanRGBFaceEmbedder:
    """Deterministic baseline for aligned 112x112 RGB face crops."""

    dim: int = 512

    def embed_aligned(self, face_rgb_112: np.ndarray) -> np.ndarray:
        if face_rgb_112.ndim != 3 or face_rgb_112.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 face RGB, got shape={face_rgb_112.shape}")
        mean = face_rgb_112.astype(np.float32).mean(axis=(0, 1))
        v = np.tile(mean, int(np.ceil(self.dim / 3)))[: self.dim].astype(np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n)


class InsightFaceArcFaceEmbedder:
    """
    Optional ArcFace embedder using insightface.

    Expects aligned 112x112 RGB crop.
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1):
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "insightface not installed; install atlas-forgery[insightface]"
            ) from e

        self._app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def embed_aligned(self, face_rgb_112: np.ndarray) -> np.ndarray:
        # insightface expects BGR images for detection pipelines; we only need embedding,
        # but the simplest stable path is to run get() on a dummy detection.
        import cv2  # pragma: no cover

        img_bgr = cv2.cvtColor(face_rgb_112, cv2.COLOR_RGB2BGR)
        faces = self._app.get(img_bgr)
        if not faces:
            raise ValueError("No face detected in aligned crop")
        v = faces[0].embedding.astype(np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n)

