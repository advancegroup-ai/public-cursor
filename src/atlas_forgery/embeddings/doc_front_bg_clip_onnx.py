from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Pixel-space bbox: (x1, y1, x2, y2) in the *same* image coordinate system."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, w: int, h: int) -> "BBox":
        x1 = int(max(0, min(self.x1, w)))
        y1 = int(max(0, min(self.y1, h)))
        x2 = int(max(0, min(self.x2, w)))
        y2 = int(max(0, min(self.y2, h)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @property
    def is_empty(self) -> bool:
        return (self.x2 - self.x1) <= 1 or (self.y2 - self.y1) <= 1


def zero_out_bbox(img_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    """Return a copy of img with bbox region zeroed out (BGR=0)."""

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image (H,W,3), got {img_bgr.shape}")
    h, w = img_bgr.shape[:2]
    bb = bbox.clamp(w=w, h=h)
    if bb.is_empty:
        return img_bgr.copy()
    out = img_bgr.copy()
    out[bb.y1 : bb.y2, bb.x1 : bb.x2] = 0
    return out


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


def _clip_preprocess_bgr(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """
    CLIP-style preprocessing.

    Notes:
    - This module is meant as a lightweight prototype. The exact mean/std + resize strategy
      must match the ONNX you use (your production model may differ).
    """

    import cv2

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, 0)  # NCHW
    return x


class ClipBgEmbedderONNX:
    """
    Doc-front background embedding (CLIP ViT-B/32-ish) with optional face masking.

    Intended contract:
    - Input: doc_front crop (ideally card boundary already detected & rectified).
    - Optional: face bbox on-card → zero it out before embedding.
    - Output: 512-dim L2-normalized vector.
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[list[str]] = None,
        input_size: int = 224,
    ) -> None:
        import onnxruntime as ort

        self.onnx_path = onnx_path
        self.input_size = int(input_size)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_opts, providers=providers or ort.get_available_providers()
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(
        self,
        img_bgr: np.ndarray,
        face_bbox: Optional[BBox] = None,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        if face_bbox is not None:
            img_bgr = zero_out_bbox(img_bgr, face_bbox)
        x = _clip_preprocess_bgr(img_bgr, size=self.input_size)
        y = self.session.run([self.output_name], {self.input_name: x})[0]
        y = np.asarray(y).reshape(-1).astype(np.float32)
        if l2_normalize:
            y = _l2_normalize(y)
        return y


def detect_face_bbox_insightface(img_bgr: np.ndarray) -> Optional[BBox]:
    """
    Best-effort face bbox detector using `insightface`.

    Returns the largest detected face bbox (x1,y1,x2,y2) in pixel coordinates.
    """

    try:
        from insightface.app import FaceAnalysis
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "insightface is required for face detection. Install with `pip install insightface`."
        ) from e

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = app.get(img_bgr)
    if not faces:
        return None
    faces = sorted(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        reverse=True,
    )
    x1, y1, x2, y2 = [int(v) for v in faces[0].bbox.tolist()]
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)
 
