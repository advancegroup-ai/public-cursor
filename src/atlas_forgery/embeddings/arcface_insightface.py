from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
    return x / denom


@dataclass(frozen=True)
class ArcFaceConfig:
    """
    InsightFace wrapper config.
    """

    model_name: str = "buffalo_l"
    det_size: Tuple[int, int] = (640, 640)
    ctx_id: int = 0  # -1 CPU


class ArcFaceEmbedder:
    """
    Face embedding via InsightFace (ArcFace-like 512-dim).

    - Detects the most prominent face (highest det_score) and returns its 512D embedding.
    - If you already have aligned 112x112, you can supply it via `embed_aligned_bgr112`.
    """

    def __init__(self, cfg: ArcFaceConfig):
        self.cfg = cfg
        self.app = FaceAnalysis(name=cfg.model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=cfg.ctx_id, det_size=cfg.det_size)

    def embed_bgr(self, img_bgr: np.ndarray, l2_normalize: bool = True) -> Optional[np.ndarray]:
        faces = self.app.get(img_bgr)
        if not faces:
            return None
        faces = sorted(faces, key=lambda f: float(getattr(f, "det_score", 0.0)), reverse=True)
        emb = np.asarray(faces[0].embedding, dtype=np.float32).reshape(-1)
        if l2_normalize:
            emb = _l2_normalize(emb)
        return emb

    def embed_aligned_bgr112(self, aligned_bgr112: np.ndarray, l2_normalize: bool = True) -> np.ndarray:
        if aligned_bgr112.shape[0] != 112 or aligned_bgr112.shape[1] != 112:
            aligned_bgr112 = cv2.resize(aligned_bgr112, (112, 112), interpolation=cv2.INTER_AREA)
        faces = self.app.get(aligned_bgr112)
        if faces:
            emb = np.asarray(faces[0].embedding, dtype=np.float32).reshape(-1)
        else:
            # As a fallback, run detection on the 112x112 crop anyway; if still no face, return zeros.
            emb = np.zeros((512,), dtype=np.float32)
        if l2_normalize:
            emb = _l2_normalize(emb)
        return emb

