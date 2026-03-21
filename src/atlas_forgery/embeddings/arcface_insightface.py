from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FaceEmbeddingResult:
    embedding: np.ndarray  # (512,)
    det_score: float
    bbox_xyxy: np.ndarray  # (4,)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


class ArcFaceEmbedder:
    """
    ArcFace 512-dim embedding using insightface's `FaceAnalysis` pipeline.

    For production parity with your "guardian CV service", you’ll likely want to swap
    the model pack and alignment logic to match exactly; this is a practical prototype.
    """

    def __init__(self, providers: Optional[list[str]] = None) -> None:
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "insightface is required for ArcFace embeddings. Install with `pip install insightface`."
            ) from e

        self._FaceAnalysis = FaceAnalysis
        self.providers = providers or ["CPUExecutionProvider"]
        self.app = self._FaceAnalysis(name="buffalo_l", providers=self.providers)
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def embed_largest_face(self, img_bgr: np.ndarray, l2_normalize: bool = True) -> Optional[FaceEmbeddingResult]:
        faces = self.app.get(img_bgr)
        if not faces:
            return None
        faces = sorted(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            reverse=True,
        )
        f = faces[0]
        emb = np.asarray(f.embedding, dtype=np.float32).reshape(-1)
        if l2_normalize:
            emb = _l2_normalize(emb)
        return FaceEmbeddingResult(
            embedding=emb,
            det_score=float(getattr(f, "det_score", 0.0)),
            bbox_xyxy=np.asarray(f.bbox, dtype=np.float32).reshape(4),
        )
