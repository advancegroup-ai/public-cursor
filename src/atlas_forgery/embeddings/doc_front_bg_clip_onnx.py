from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
    return x / denom


def _center_crop_square(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img_bgr[y0 : y0 + s, x0 : x0 + s]


def _resize(img_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)


def mask_face_region(img_bgr: np.ndarray, face_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Zero out the face area in-place (copy) using xyxy pixels.
    Matches the referenced behavior: img[y1:y2, x1:x2] = 0
    """
    x1, y1, x2, y2 = face_xyxy
    x1 = max(0, min(int(x1), img_bgr.shape[1]))
    x2 = max(0, min(int(x2), img_bgr.shape[1]))
    y1 = max(0, min(int(y1), img_bgr.shape[0]))
    y2 = max(0, min(int(y2), img_bgr.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return img_bgr
    out = img_bgr.copy()
    out[y1:y2, x1:x2] = 0
    return out


@dataclass(frozen=True)
class ClipOnnxConfig:
    model_path: Path
    input_size: int = 224
    mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


class DocFrontBackgroundClipOnnxEmbedder:
    """
    Portable ONNX embedder for doc-front background.

    Notes:
    - This module does NOT perform card boundary detection or face detection.
      You must provide an already-cropped doc-front card image, and optionally a face bbox.
    - Preprocessing is CLIP-like: BGR->RGB, resize/crop, normalize, NCHW float32.
    """

    def __init__(self, cfg: ClipOnnxConfig, providers: Optional[list[str]] = None):
        self.cfg = cfg
        sess_opts = ort.SessionOptions()
        model_path = str(cfg.model_path)
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        img_bgr = _center_crop_square(img_bgr)
        img_bgr = _resize(img_bgr, self.cfg.input_size)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array(self.cfg.mean, dtype=np.float32)
        std = np.array(self.cfg.std, dtype=np.float32)
        img_rgb = (img_rgb - mean) / std
        x = np.transpose(img_rgb, (2, 0, 1))[None, ...]  # 1x3xHxW
        return x.astype(np.float32)

    def embed_bgr(
        self,
        img_bgr: np.ndarray,
        face_xyxy: Optional[Tuple[int, int, int, int]] = None,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        if face_xyxy is not None:
            img_bgr = mask_face_region(img_bgr, face_xyxy)
        x = self.preprocess(img_bgr)
        y = self.sess.run([self.output_name], {self.input_name: x})[0]
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if l2_normalize:
            y = _l2_normalize(y)
        return y

