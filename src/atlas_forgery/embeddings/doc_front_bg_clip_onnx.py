from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class FaceBBox:
     x0: int
     y0: int
     x1: int
     y1: int


def _to_rgb_uint8(img_bgr: np.ndarray) -> np.ndarray:
     if img_bgr is None or img_bgr.size == 0:
         raise ValueError("empty image")
     if img_bgr.dtype != np.uint8:
         img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
     return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _center_crop_resize(img_rgb: np.ndarray, size: int) -> np.ndarray:
     h, w = img_rgb.shape[:2]
     s = min(h, w)
     y0 = (h - s) // 2
     x0 = (w - s) // 2
     crop = img_rgb[y0 : y0 + s, x0 : x0 + s]
     resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
     return resized


def _detect_face_bbox_haar(img_bgr: np.ndarray) -> Optional[FaceBBox]:
     # Fallback-only: avoids extra model dependencies; OK for a crude mask.
     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
     cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
     faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
     if len(faces) == 0:
         return None
     x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
     return FaceBBox(int(x), int(y), int(x + w), int(y + h))


class DocFrontBgClipOnnxEmbedder:
     """
     Background embedding using a CLIP-style ONNX model.
 
     Notes:
     - Input preprocessing here is conservative (center-crop to square, resize to 224, normalize).
     - Face masking uses a simple Haar cascade as a portable fallback; caller can provide their own bbox.
     """
 
     def __init__(self, onnx_path: str, providers: Optional[list[str]] = None):
         self.onnx_path = str(onnx_path)
         self.providers = providers or ["CPUExecutionProvider"]
         self.sess = ort.InferenceSession(self.onnx_path, providers=self.providers)
         self.input_name = self.sess.get_inputs()[0].name
         self.output_name = self.sess.get_outputs()[0].name
 
     def embed(
         self,
         image_path: Path,
         *,
         mask_face: bool = False,
         face_bbox: Optional[FaceBBox] = None,
     ) -> np.ndarray:
         img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
         if img_bgr is None:
             raise FileNotFoundError(f"failed to read image: {image_path}")
 
         if mask_face:
             bb = face_bbox or _detect_face_bbox_haar(img_bgr)
             if bb is not None:
                 x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
                 x0 = max(0, min(x0, img_bgr.shape[1] - 1))
                 x1 = max(0, min(x1, img_bgr.shape[1]))
                 y0 = max(0, min(y0, img_bgr.shape[0] - 1))
                 y1 = max(0, min(y1, img_bgr.shape[0]))
                 if x1 > x0 and y1 > y0:
                     img_bgr[y0:y1, x0:x1] = 0
 
         img_rgb = _to_rgb_uint8(img_bgr)
         img_rgb = _center_crop_resize(img_rgb, 224)
 
         x = img_rgb.astype(np.float32) / 255.0
         mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
         std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
         x = (x - mean) / std
         x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
 
         y = self.sess.run([self.output_name], {self.input_name: x})[0]
         y = np.asarray(y).reshape(-1).astype(np.float32)
         if y.shape[0] != 512:
             raise ValueError(f"expected 512-d embedding, got shape {y.shape}")
         y = y / (np.linalg.norm(y) + 1e-12)
         return y
