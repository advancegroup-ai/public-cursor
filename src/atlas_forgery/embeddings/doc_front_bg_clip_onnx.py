from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


BBoxXYXY = Tuple[int, int, int, int]


def _to_rgb_pil(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    # Resize shortest side to size, then center-crop.
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Empty image")
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left = max(0, (nw - size) // 2)
    top = max(0, (nh - size) // 2)
    return img.crop((left, top, left + size, top + size))


def _mask_bbox_rgb(np_img: np.ndarray, bbox_xyxy: BBoxXYXY, value: int = 0) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    h, w = np_img.shape[:2]
    x1 = int(np.clip(x1, 0, w))
    x2 = int(np.clip(x2, 0, w))
    y1 = int(np.clip(y1, 0, h))
    y2 = int(np.clip(y2, 0, h))
    if x2 <= x1 or y2 <= y1:
        return np_img
    out = np_img.copy()
    out[y1:y2, x1:x2, :] = value
    return out


@dataclass(frozen=True)
class ClipOnnxPreprocess:
    input_size: int = 224
    mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073)
    std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711)

    def __call__(self, img: Image.Image) -> np.ndarray:
        img = _resize_center_crop(_to_rgb_pil(img), self.input_size)
        x = np.asarray(img).astype(np.float32) / 255.0  # HWC RGB
        x = (x - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))  # CHW
        return x[None, ...]  # NCHW


class DocFrontBackgroundEmbedderOnnx:
    """
    Portable ONNX embedder for doc-front card background templates.

    This implements the "face exclusion" step by zero-masking a provided face bbox
    before embedding. Card-boundary detection and face detection are *not* included
    here (those are platform-specific); pass bboxes from your detector.
    """

    def __init__(
        self,
        onnx_path: str,
        *,
        providers: Optional[Sequence[str]] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        preprocess: Optional[ClipOnnxPreprocess] = None,
        l2_normalize: bool = True,
    ) -> None:
        self.onnx_path = onnx_path
        self.preprocess = preprocess or ClipOnnxPreprocess()
        self.l2_normalize = l2_normalize
        sess_opts = ort.SessionOptions()
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=list(providers) if providers else None,
        )
        ins = self.session.get_inputs()
        outs = self.session.get_outputs()
        self.input_name = input_name or ins[0].name
        self.output_name = output_name or outs[0].name

    def embed_pil(self, img: Image.Image, *, face_bbox_xyxy: Optional[BBoxXYXY] = None) -> np.ndarray:
        img = _to_rgb_pil(img)
        if face_bbox_xyxy is not None:
            np_img = np.asarray(img).copy()
            np_img = _mask_bbox_rgb(np_img, face_bbox_xyxy, value=0)
            img = Image.fromarray(np_img, mode="RGB")
        x = self.preprocess(img)
        y = self.session.run([self.output_name], {self.input_name: x})[0]
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if self.l2_normalize:
            n = float(np.linalg.norm(y) + 1e-12)
            y = y / n
        return y

    def embed_path(self, image_path: str, *, face_bbox_xyxy: Optional[BBoxXYXY] = None) -> np.ndarray:
        with Image.open(image_path) as img:
            return self.embed_pil(img, face_bbox_xyxy=face_bbox_xyxy)

