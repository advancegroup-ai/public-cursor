from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


class FaceEmbedder:
    dim: int

    def embed_aligned_rgb112(self, face_rgb112: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def embed_path(self, image_path: str | Path) -> np.ndarray:
        img = Image.open(image_path).convert("RGB").resize((112, 112))
        arr = np.asarray(img)
        return self.embed_aligned_rgb112(arr)


@dataclass(frozen=True)
class DeterministicFaceEmbedder(FaceEmbedder):
    """
    Deterministic baseline: mean RGB of 112x112 then pads to dim.
    """

    dim: int = 512

    def embed_aligned_rgb112(self, face_rgb112: np.ndarray) -> np.ndarray:
        x = np.asarray(face_rgb112, dtype=np.float32)
        m = x.reshape(-1, 3).mean(axis=0)
        out = np.zeros((self.dim,), dtype=np.float32)
        out[:3] = m
        n = np.linalg.norm(out)
        if n > 0:
            out /= n
        return out


class InsightFaceArcFaceEmbedder(FaceEmbedder):
    """
    Optional ArcFace wrapper via insightface.
    This assumes the provided face crop is already aligned to 112x112.
    """

    def __init__(self, *, model_name: str = "buffalo_l", dim: int = 512) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "insightface is required (pip install atlas-forgery[insightface])"
            ) from e

        self.dim = int(dim)
        self._app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    def embed_aligned_rgb112(self, face_rgb112: np.ndarray) -> np.ndarray:  # pragma: no cover
        # InsightFace expects BGR images in many code paths; keep explicit conversion here.
        rgb = np.asarray(face_rgb112, dtype=np.uint8)
        bgr = rgb[..., ::-1]
        faces = self._app.get(bgr)
        if not faces:
            raise ValueError("No face detected in aligned crop")
        emb = np.asarray(faces[0].embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {emb.shape[0]}")
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        return emb
