from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ArcFaceResult:
    embedding: np.ndarray  # shape (512,), float32
    bbox_xyxy: Optional[tuple[int, int, int, int]] = None
    det_score: Optional[float] = None


class ArcFaceInsightFaceEmbedder:
    """
    Minimal ArcFace embedding wrapper using InsightFace if installed.

    Notes:
    - This class expects BGR images as numpy arrays (OpenCV convention).
    - It runs face detection + embedding using InsightFace FaceAnalysis.
    """

    def __init__(self, det_size: tuple[int, int] = (640, 640), ctx_id: int = -1) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "insightface is required for ArcFaceInsightFaceEmbedder. "
                "Install insightface (and its deps) in your environment."
            ) from e

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def embed_best_face_bgr(self, img_bgr: np.ndarray, l2_normalize: bool = True) -> ArcFaceResult:
        faces = self.app.get(img_bgr)
        if not faces:
            raise ValueError("No face detected")

        best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        emb = np.asarray(best.embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != 512:
            raise ValueError(f"Unexpected embedding dim: {emb.shape}")

        if l2_normalize:
            n = float(np.linalg.norm(emb) + 1e-12)
            emb = emb / n

        bbox = None
        if getattr(best, "bbox", None) is not None:
            x0, y0, x1, y1 = best.bbox.astype(int).tolist()
            bbox = (x0, y0, x1, y1)

        return ArcFaceResult(
            embedding=emb,
            bbox_xyxy=bbox,
            det_score=float(getattr(best, "det_score", 0.0)),
        )

