from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..masking import mask_bbox_zero_rgb
from ..types import BBoxXYXY


class DocEmbedder(Protocol):
    def embed(self, img_rgb: np.ndarray) -> np.ndarray:  # (512,)
        ...


@dataclass(frozen=True)
class MeanRGBEmbedder:
    """
    Deterministic baseline embedder for testing and pipelines.

    Produces a 512-d vector by repeating the per-channel mean.
    """

    dim: int = 512

    def embed(self, img_rgb: np.ndarray) -> np.ndarray:
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape={img_rgb.shape}")
        mean = img_rgb.astype(np.float32).mean(axis=(0, 1))  # (3,)
        v = np.tile(mean, int(np.ceil(self.dim / 3)))[: self.dim].astype(np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n)


@dataclass(frozen=True)
class MaskedDocEmbedder:
    """
    Wraps a DocEmbedder by zero-masking a face bbox on the document.
    """

    embedder: DocEmbedder

    def embed(self, img_rgb: np.ndarray, face_bbox_xyxy: BBoxXYXY | None) -> np.ndarray:
        masked = img_rgb if face_bbox_xyxy is None else mask_bbox_zero_rgb(img_rgb, face_bbox_xyxy)
        return self.embedder.embed(masked)


class OnnxClipEmbedder:
    """
    Optional ONNX-backed CLIP embedder (ViT-B/32 style output 512).

    Requires onnxruntime at runtime. Input is RGB uint8 HxWx3.
    """

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except Exception as e:  # pragma: no cover
            raise RuntimeError("onnxruntime not installed; install atlas-forgery[onnx]") from e

        self._ort = ort
        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

    def embed(self, img_rgb: np.ndarray) -> np.ndarray:
        if img_rgb.dtype != np.uint8:
            img_rgb = img_rgb.astype(np.uint8)
        # Common simple convention: NCHW float32 in [0,1]
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xHxW
        y = self._sess.run([self._output_name], {self._input_name: x})[0]
        v = np.asarray(y).reshape(-1).astype(np.float32)
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n)

