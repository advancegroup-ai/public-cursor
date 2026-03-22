from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


def mask_bbox_zero_rgb(img: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """
    Zero out pixels inside bbox on an RGB uint8 image.
    bbox_xyxy: (x0, y0, x1, y1) with x1/y1 exclusive-like.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3); got shape={img.shape}")
    x0, y0, x1, y1 = bbox_xyxy
    h, w = img.shape[:2]
    x0 = int(max(0, min(w, x0)))
    x1 = int(max(0, min(w, x1)))
    y0 = int(max(0, min(h, y0)))
    y1 = int(max(0, min(h, y1)))
    if x1 <= x0 or y1 <= y0:
        return img
    out = img.copy()
    out[y0:y1, x0:x1, :] = 0
    return out


class DocEmbedder(Protocol):
    dim: int

    def embed(self, rgb_u8: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class MeanRGBEmbedder:
    dim: int = 3

    def embed(self, rgb_u8: np.ndarray) -> np.ndarray:
        x = np.asarray(rgb_u8, dtype=np.float32)
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3); got shape={x.shape}")
        v = x.reshape(-1, 3).mean(axis=0)
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype(np.float32)


@dataclass(frozen=True)
class OnnxClipEmbedder:
    """
    Minimal ONNXRuntime wrapper for a CLIP-like image embedding model.

    Expected model IO:
    - input: float32 tensor NCHW in [0,1], shape (1,3,H,W)
    - output: float32 embedding shape (1,512) or (512,)
    """

    onnx_path: str
    dim: int = 512
    input_name: str | None = None
    output_name: str | None = None

    def _session(self):
        import onnxruntime as ort

        return ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])

    def embed(self, rgb_u8: np.ndarray) -> np.ndarray:
        sess = self._session()
        input_name = self.input_name or sess.get_inputs()[0].name
        output_name = self.output_name or sess.get_outputs()[0].name

        x = np.asarray(rgb_u8, dtype=np.uint8)
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3); got shape={x.shape}")
        # Convert to float NCHW in [0,1]
        x_f = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        out = sess.run([output_name], {input_name: x_f})[0]
        v = np.asarray(out, dtype=np.float32).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"Expected dim={self.dim}; got {v.size}")
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype(np.float32)


@dataclass(frozen=True)
class MaskedDocEmbedder:
    embedder: DocEmbedder

    def embed(self, rgb_u8: np.ndarray, face_bbox_xyxy: tuple[int, int, int, int] | None) -> np.ndarray:
        x = np.asarray(rgb_u8, dtype=np.uint8)
        if face_bbox_xyxy is not None:
            x = mask_bbox_zero_rgb(x, face_bbox_xyxy)
        return self.embedder.embed(x)


def load_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)

