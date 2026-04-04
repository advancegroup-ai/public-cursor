from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np
from PIL import Image


BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


class OnnxLikeSession(Protocol):
    def run(self, output_names, input_feed): ...


def _clip_preprocess(img: Image.Image, image_size: int = 224) -> np.ndarray:
    """
    Minimal CLIP-style preprocessing:
    - resize shortest side to image_size (keeping aspect)
    - center crop to image_size x image_size
    - convert to RGB float32
    - normalize by CLIP mean/std
    Output: (1, 3, H, W)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = image_size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    left = max(0, (new_w - image_size) // 2)
    top = max(0, (new_h - image_size) // 2)
    img = img.crop((left, top, left + image_size, top + image_size))

    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, RGB
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std
    return arr[None, ...]  # NCHW


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(vec) + eps
    return vec / denom


def _mask_face_bbox_rgb(img: Image.Image, face_bbox: BBox) -> Image.Image:
    x1, y1, x2, y2 = face_bbox
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c > x1c and y2c > y1c:
        arr[y1c:y2c, x1c:x2c, :] = 0
    return Image.fromarray(arr, mode="RGB")


@dataclass(frozen=True)
class DocFrontBackgroundEmbedder:
    """
    Doc-front background embedding with optional face exclusion masking.

    The ONNX model is expected to return a 512-d embedding for a single image.
    """

    session: OnnxLikeSession
    input_name: str = "input"
    output_name: Optional[str] = None
    l2_normalize: bool = True

    def embed_pil(self, img: Image.Image, face_bbox: Optional[BBox] = None) -> np.ndarray:
        if face_bbox is not None:
            img = _mask_face_bbox_rgb(img, face_bbox)

        x = _clip_preprocess(img)  # (1,3,224,224)
        outs = self.session.run([self.output_name] if self.output_name else None, {self.input_name: x})
        if not outs:
            raise RuntimeError("ONNX session returned no outputs")

        vec = np.asarray(outs[0]).reshape(-1).astype(np.float32)
        if vec.shape[0] != 512:
            raise ValueError(f"Expected 512-d embedding, got shape {vec.shape}")
        if self.l2_normalize:
            vec = _l2_normalize(vec)
        return vec

