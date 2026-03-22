from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class DeterministicHasherEmbedder:
    """
    Deterministic baseline embedder (no ML deps) for plumbing/tests.

    Produces a fixed-dim vector from raw bytes. Not meaningful semantically.
    """
 
    dim: int = 512

    def embed(self, x: np.ndarray) -> np.ndarray:
        b = np.ascontiguousarray(x).view(np.uint8).tobytes()
        v = np.zeros((self.dim,), dtype=np.float32)
        h = 2166136261
        for i, bb in enumerate(b):
            h = (h ^ bb) * 16777619 & 0xFFFFFFFF
            v[(h + i) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / (n + 1e-12)


@dataclass(frozen=True, slots=True)
class OnnxClipImageEmbedder:
    """
    Thin wrapper around an ONNX CLIP-style image embedding model.

    This repo does not vendor weights. Provide a model path at runtime.
    """
 
    model_path: str | Path
    dim: int = 512
    providers: tuple[str, ...] = ("CPUExecutionProvider",)

    def _session(self):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime not installed. Install with `pip install forgery-detection[onnx]`."
            ) from e
        return ort.InferenceSession(str(self.model_path), providers=list(self.providers))

    def embed(self, x: np.ndarray) -> np.ndarray:
        """
        x: image as np.ndarray (HWC uint8/float). Preprocessing depends on model.

        For real pipelines, adapt preprocessing to match your exported ONNX graph.
        """
        sess = self._session()
        inputs = sess.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"expected 1 input, got {len(inputs)}")

        inp_name = inputs[0].name
        img = x.astype(np.float32, copy=False)
        if img.ndim != 3:
            raise ValueError(f"expected HWC image, got shape {img.shape}")
        img = np.transpose(img, (2, 0, 1))[None, ...]
        if img.max() > 1.5:
            img = img / 255.0

        out = sess.run(None, {inp_name: img})[0]
        out = np.asarray(out).reshape(-1).astype(np.float32)
        if out.size != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {out.size}")
        n = np.linalg.norm(out)
        return out / (n + 1e-12)
