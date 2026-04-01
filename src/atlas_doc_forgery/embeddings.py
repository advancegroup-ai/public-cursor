from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from .masking import mask_face_on_document


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = float(np.linalg.norm(x)) + eps
    return x / denom


@dataclass(frozen=True)
class EmbedConfig:
    clip_onnx_path: str
    clip_input_size: int = 224
    providers: Optional[list[str]] = None


class ClipOnnxEmbedder:
    def __init__(self, cfg: EmbedConfig) -> None:
        providers = cfg.providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(cfg.clip_onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = int(cfg.clip_input_size)

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(image_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess(image_bgr)
        y = self.session.run(None, {self.input_name: x})[0]
        v = y.reshape(-1).astype(np.float32)
        return _l2norm(v)


class ArcFaceEmbedder:
    def __init__(self, model_name: str = "buffalo_l", providers: Optional[list[str]] = None) -> None:
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            raise RuntimeError("insightface is required for ArcFace embedding") from e

        self.app = FaceAnalysis(name=model_name, providers=providers or ["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        faces = self.app.get(image_bgr)
        if not faces:
            raise ValueError("No face detected")
        face = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[0]
        emb = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
        return _l2norm(emb)


def embed_doc_front_background(
    image_bgr: np.ndarray,
    clip: ClipOnnxEmbedder,
    return_meta: bool = False,
):
    masked = mask_face_on_document(image_bgr)
    emb = clip.embed(masked.masked_image_bgr)
    if return_meta:
        return emb, {
            "doc_bbox": masked.doc_bbox,
            "face_bbox_on_doc": masked.face_bbox_on_doc,
            "masked": masked.face_bbox_on_doc is not None,
        }
    return emb
