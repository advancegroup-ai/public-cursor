from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from PIL import Image

from .image_utils import BBox, load_rgb_image


class FaceEmbeddingBackend(Protocol):
    def embed_aligned_112(self, face_rgb_112: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Input: RGB uint8 (112,112,3); Output: float32 (512,)"""
        ...


@dataclass(frozen=True)
class ArcFaceConfig:
    size: int = 112
    l2_normalize: bool = True


class ArcFaceEmbedder:
    """
    Lightweight wrapper around an ArcFace-like backend.

    This repo intentionally does not vendor weights. You can supply a backend
    that knows how to run the actual model (e.g. InsightFace, ONNXRuntime, etc).
    """

    def __init__(self, backend: FaceEmbeddingBackend, config: ArcFaceConfig = ArcFaceConfig()) -> None:
        self._backend = backend
        self._cfg = config

    def embed(self, image: str | Image.Image, face_bbox: Optional[BBox] = None) -> np.ndarray:
        pil = load_rgb_image(image)
        if face_bbox is not None:
            w, h = pil.size
            b = face_bbox.clamp(w, h)
            if b.is_empty():
                raise ValueError("face_bbox is empty after clamping")
            pil = pil.crop((b.x1, b.y1, b.x2, b.y2))
        pil = pil.resize((self._cfg.size, self._cfg.size), resample=Image.BICUBIC)

        face_rgb = np.asarray(pil).astype(np.uint8)
        if face_rgb.shape != (self._cfg.size, self._cfg.size, 3):
            raise ValueError(f"Expected {(self._cfg.size, self._cfg.size, 3)}, got {face_rgb.shape}")

        vec = np.asarray(self._backend.embed_aligned_112(face_rgb)).reshape(-1).astype(np.float32)
        if vec.shape[0] != 512:
            raise ValueError(f"Expected 512-dim face embedding, got shape={vec.shape}")
        if self._cfg.l2_normalize:
            n = float(np.linalg.norm(vec) + 1e-12)
            vec = vec / n
        return vec

