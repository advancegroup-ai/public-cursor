from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ArcFaceInsightFaceEmbedder:
    """
    512D face embedding via InsightFace.

    Input expectation: a face image (RGB/BGR ok) containing a single face. If you want
    aligned faces, do alignment upstream; this class focuses on embedding.
    """

    model_name: str = "buffalo_l"
    det_size: tuple[int, int] = (640, 640)
    ctx_id: int = -1  # -1: CPU, 0+: GPU
    output_l2_normalize: bool = True

    def __post_init__(self) -> None:
        # Lazy import to keep base install lighter.
        from insightface.app import FaceAnalysis  # type: ignore

        app = FaceAnalysis(name=self.model_name, providers=None)
        app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        object.__setattr__(self, "_app", app)

    def embed_bgr(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        faces = self._app.get(img_bgr)
        if not faces:
            return None
        # Choose the largest detected face.
        areas = []
        for f in faces:
            x1, y1, x2, y2 = map(float, f.bbox)
            areas.append((x2 - x1) * (y2 - y1))
        f = faces[int(np.argmax(areas))]
        emb = np.asarray(f.embedding, dtype=np.float32).reshape(-1)
        if self.output_l2_normalize:
            emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb

