from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .masking import BBoxXYXY, mask_bbox_zero_rgb
from .similarity import l2_normalize


class DocEmbedder(Protocol):
    def embed_rgb(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        """Return float32 vector of shape (D,)."""


@dataclass(frozen=True)
class MeanRGBEmbedder:
    """Deterministic baseline embedder for tests / environments without models."""

    out_dim: int = 512

    def embed_rgb(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        x = img_rgb_u8.astype(np.float32) / 255.0
        mean = x.mean(axis=(0, 1))  # (3,)
        v = np.zeros((self.out_dim,), dtype=np.float32)
        v[:3] = mean
        return l2_normalize(v)


@dataclass(frozen=True)
class MaskedDocEmbedder:
    base: DocEmbedder
    face_bbox: BBoxXYXY | None = None

    def embed_rgb(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        img2 = img_rgb_u8
        if self.face_bbox is not None:
            img2 = mask_bbox_zero_rgb(img_rgb_u8, self.face_bbox)
        return self.base.embed_rgb(img2)


@dataclass
class OnnxClipEmbedder:
    """
    CLIP-like embedding via ONNXRuntime.

    Notes:
    - This intentionally keeps preprocessing minimal and model-agnostic.
    - For production parity, wire this to the exact preprocessing used by your ONNX export.
    """

    onnx_path: str | Path
    input_name: str | None = None
    output_name: str | None = None

    def __post_init__(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("onnxruntime is required for OnnxClipEmbedder") from e

        sess_opts = ort.SessionOptions()
        self._session = ort.InferenceSession(str(self.onnx_path), sess_options=sess_opts)
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if self.input_name is None:
            self.input_name = inputs[0].name
        if self.output_name is None:
            self.output_name = outputs[0].name

    def embed_rgb(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        # Generic: NCHW float32 in [0,1]
        x = img_rgb_u8.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,H,W)
        out = self._session.run([self.output_name], {self.input_name: x})[0]
        v = np.asarray(out).reshape(-1).astype(np.float32, copy=False)
        return l2_normalize(v)

