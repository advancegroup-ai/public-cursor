from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


try:
    from insightface.app import FaceAnalysis
except Exception as e:  # pragma: no cover
    FaceAnalysis = None  # type: ignore[assignment]
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


BBoxXYXY = Tuple[int, int, int, int]


@dataclass(frozen=True)
class InsightFaceArcFaceEmbedder:
    """
    512D face embedding using InsightFace's ArcFace models.

    - If you already have aligned 112x112 face crops, pass them directly via `embed_aligned`.
    - If you only have full images, use `embed_bgr` which runs detection+alignment.
    """

    det_size: Tuple[int, int] = (640, 640)
    providers: Optional[Sequence[str]] = None
    ctx_id: int = -1  # -1=CPU, 0+=GPU index (InsightFace)
    name: str = "buffalo_l"  # common pack containing detection+recognition

    def _app(self) -> "FaceAnalysis":
        if FaceAnalysis is None:  # pragma: no cover
            raise ImportError(f"insightface import failed: {_IMPORT_ERR}")
        app = FaceAnalysis(name=self.name, providers=list(self.providers) if self.providers else None)
        app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        return app

    def embed_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Detect largest face in image and return 512D embedding (L2-normalized by InsightFace).
        """
        app = self._app()
        faces = app.get(img_bgr)
        if not faces:
            raise ValueError("No face detected")
        faces = sorted(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])), reverse=True)
        emb = faces[0].normed_embedding
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    def embed_aligned(self, aligned_112x112_bgr: np.ndarray) -> np.ndarray:
        """
        Compute embedding from an already-aligned 112x112 face crop (BGR).
        """
        app = self._app()
        # Use recognition model directly when available via app.models['recognition']
        rec = app.models.get("recognition")
        if rec is None:
            raise RuntimeError("InsightFace recognition model not available")
        emb = rec.get_feat(aligned_112x112_bgr)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(emb) + 1e-12)
        return emb / n

