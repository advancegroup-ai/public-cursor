from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = float(np.linalg.norm(vec)) + eps
    return (vec / denom).astype(np.float32, copy=False)


class FaceEmbedder(Protocol):
    def embed(self, image: np.ndarray) -> np.ndarray:
        ...


@dataclass
class DeterministicFaceEmbedder:
    dim: int = 512

    def embed(self, image: np.ndarray) -> np.ndarray:
        arr = image.astype(np.float32)
        h, w = arr.shape[:2]
        rgb_means = arr.reshape(-1, 3).mean(axis=0)
        base = np.array([h, w, rgb_means[0], rgb_means[1], rgb_means[2]], dtype=np.float32)
        tiled = np.resize(base, self.dim).astype(np.float32, copy=False)
        return l2_normalize(tiled)


class InsightFaceArcFaceEmbedder:
    def __init__(self, det_size: tuple[int, int] = (640, 640), providers: list[str] | None = None):
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "insightface is required for InsightFaceArcFaceEmbedder. "
                "Install with: pip install '.[face]'"
            ) from exc

        self._app = FaceAnalysis(name="buffalo_l", providers=providers or ["CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=det_size)

    def embed(self, image: np.ndarray) -> np.ndarray:
        faces = self._app.get(image)
        if not faces:
            raise ValueError("No face detected in image")
        best = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
        vec = np.asarray(best.embedding, dtype=np.float32)
        if vec.shape[0] != 512:
            raise ValueError(f"Unexpected embedding dim: {vec.shape[0]} (expected 512)")
        return l2_normalize(vec)


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))
