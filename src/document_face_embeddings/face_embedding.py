from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedding import l2_normalize


class FaceEmbedder:
    dim: int

    def embed_face_bgr(self, face_bgr: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class DeterministicFaceEmbedder(FaceEmbedder):
    """Stable fallback when ArcFace deps/models aren't available.

    Uses normalized per-channel mean on BGR input -> 3D vector.
    """

    dim: int = 3

    def embed_face_bgr(self, face_bgr: np.ndarray) -> np.ndarray:
        x = np.asarray(face_bgr, dtype=np.float32)
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError("Expected HxWx3 BGR face image")
        v = x.mean(axis=(0, 1)) / 255.0
        return l2_normalize(v)


class InsightFaceArcFaceEmbedder(FaceEmbedder):
    """ArcFace 512-dim embedder via insightface (optional dependency).

    Notes:
    - Expects an already-aligned 112x112 face crop (BGR).
    - Uses insightface model_zoo; the exact model can be swapped as needed.
    """

    dim: int = 512

    def __init__(self, model_name: str = "buffalo_l") -> None:
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "insightface is required for InsightFaceArcFaceEmbedder. "
                "Install extras: pip install 'document-face-embeddings[arcface]'"
            ) from e

        self._app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    def embed_face_bgr(self, face_bgr: np.ndarray) -> np.ndarray:
        # insightface FaceAnalysis expects full images and runs detection; for aligned crops
        # we still run it and take the largest face. This is a pragmatic default.
        faces = self._app.get(face_bgr)
        if not faces:
            raise ValueError("No face detected in provided crop")
        face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        emb = np.asarray(face.embedding, dtype=np.float32)
        if emb.shape != (512,):
            raise ValueError(f"Unexpected embedding shape: {emb.shape}")
        return l2_normalize(emb)

