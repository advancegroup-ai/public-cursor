from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class ArcFaceResult:
     embedding: np.ndarray
     bbox_xyxy: tuple[int, int, int, int]


class ArcFaceInsightFaceEmbedder:
     """
     ArcFace-style 512D face embedding using InsightFace.
 
     Uses `insightface.app.FaceAnalysis` which performs detection + alignment internally.
     """
 
     def __init__(self, det_size: tuple[int, int] = (640, 640), providers: Optional[list[str]] = None):
         from insightface.app import FaceAnalysis
 
         self.providers = providers
         self.app = FaceAnalysis(name="buffalo_l", providers=providers)
         self.app.prepare(ctx_id=0, det_size=det_size)
 
     def embed(self, image_path: Path) -> np.ndarray:
         res = self.embed_with_meta(image_path)
         return res.embedding
 
     def embed_with_meta(self, image_path: Path) -> ArcFaceResult:
         img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
         if img_bgr is None:
             raise FileNotFoundError(f"failed to read image: {image_path}")
 
         faces = self.app.get(img_bgr)
         if not faces:
             raise RuntimeError(f"no face detected in {image_path}")
         f = sorted(faces, key=lambda x: float((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])), reverse=True)[0]
 
         emb = np.asarray(f.embedding, dtype=np.float32).reshape(-1)
         if emb.shape[0] != 512:
             raise ValueError(f"expected 512-d face embedding, got shape {emb.shape}")
         emb = emb / (np.linalg.norm(emb) + 1e-12)
 
         x0, y0, x1, y1 = [int(v) for v in f.bbox]
         return ArcFaceResult(embedding=emb, bbox_xyxy=(x0, y0, x1, y1))
