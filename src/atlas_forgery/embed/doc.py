from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..mask import BBoxXYXY, mask_bbox_zero_rgb


class DocEmbedder:
    dim: int

    def embed_rgb(self, image_rgb: np.ndarray) -> np.ndarray:  # [dim]
        raise NotImplementedError

    def embed_path(self, image_path: str | Path) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img)
        return self.embed_rgb(arr)


@dataclass(frozen=True)
class MeanRGBEmbedder(DocEmbedder):
    """
    Deterministic baseline embedder: returns mean RGB then pads/truncates to dim.
    Useful for tests and wiring validation when real CLIP is unavailable.
    """

    dim: int = 512

    def embed_rgb(self, image_rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(image_rgb, dtype=np.float32)
        m = x.reshape(-1, 3).mean(axis=0)  # [3]
        out = np.zeros((self.dim,), dtype=np.float32)
        out[:3] = m
        n = np.linalg.norm(out)
        if n > 0:
            out /= n
        return out


@dataclass(frozen=True)
class MaskedDocEmbedder(DocEmbedder):
    base: DocEmbedder
    face_bbox: BBoxXYXY | None = None

    @property
    def dim(self) -> int:
        return int(self.base.dim)

    def embed_rgb(self, image_rgb: np.ndarray) -> np.ndarray:
        arr = image_rgb
        if self.face_bbox is not None:
            arr = mask_bbox_zero_rgb(arr, self.face_bbox)
        return self.base.embed_rgb(arr)


class OnnxClipEmbedder(DocEmbedder):
    """
    Optional ONNX CLIP-style embedder wrapper.
    Expects an ONNX model that takes an RGB image tensor and returns a 512-d vector.
    The exact pre/post processing is model-specific; wire this to your internal model.
    """

    def __init__(self, onnx_path: str | Path, *, dim: int = 512) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("onnxruntime is required (pip install atlas-forgery[onnx])") from e

        self.dim = int(dim)
        self._sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

    def embed_rgb(self, image_rgb: np.ndarray) -> np.ndarray:  # pragma: no cover
        x = np.asarray(image_rgb, dtype=np.float32)
        # Default: normalize to [0,1] and add batch dimension.
        x = x / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        x = x[None, ...]  # 1CHW
        out = self._sess.run([self._output_name], {self._input_name: x})[0]
        vec = np.asarray(out, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vec.shape[0]}")
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec
