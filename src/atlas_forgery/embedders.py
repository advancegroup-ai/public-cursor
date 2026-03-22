from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .masking import BBoxXYXY, mask_bbox_zero_rgb


class DocEmbedder(Protocol):
    dim: int

    def embed_rgb(self, img_rgb: np.ndarray) -> np.ndarray:  # (D,)
        ...


@dataclass(frozen=True)
class MeanRGBEmbedder:
    """
    Deterministic baseline: mean RGB expanded to a fixed dim.
    Useful for testing pipelines without heavy model deps.
    """

    dim: int = 512

    def embed_rgb(self, img_rgb: np.ndarray) -> np.ndarray:
        mean_rgb = img_rgb.astype(np.float32).mean(axis=(0, 1)) / 255.0  # (3,)
        v = np.tile(mean_rgb, int(np.ceil(self.dim / 3.0)))[: self.dim]
        return v.astype(np.float32)


@dataclass(frozen=True)
class MaskedDocEmbedder:
    base: DocEmbedder

    @property
    def dim(self) -> int:
        return int(self.base.dim)

    def embed_rgb_with_face_bbox(
        self,
        img_rgb: np.ndarray,
        face_bbox: BBoxXYXY | None,
    ) -> np.ndarray:
        if face_bbox is None:
            return self.base.embed_rgb(img_rgb)
        masked = mask_bbox_zero_rgb(img_rgb, face_bbox)
        return self.base.embed_rgb(masked)


@dataclass(frozen=True)
class OnnxClipEmbedder:
    """
    ONNX CLIP-based background embedding.

    Expects an ONNX model that takes an image tensor and outputs a 512-d vector.
    This is a lightweight wrapper; actual input/output names are introspected.
    """

    model_path: Path
    providers: tuple[str, ...] = ("CPUExecutionProvider",)

    dim: int = 512

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))

    def _session(self):
        import onnxruntime as ort

        so = ort.SessionOptions()
        return ort.InferenceSession(
            str(self.model_path),
            sess_options=so,
            providers=list(self.providers),
        )

    def embed_rgb(self, img_rgb: np.ndarray) -> np.ndarray:
        sess = self._session()
        inp = sess.get_inputs()[0].name
        out0 = sess.get_outputs()[0].name

        # Minimal preprocessing: resize to 224, normalize to [0,1], NCHW float32.
        pil = Image.fromarray(img_rgb, mode="RGB").resize((224, 224), Image.BILINEAR)
        x = np.asarray(pil).astype(np.float32) / 255.0  # HWC
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3x224x224

        y = sess.run([out0], {inp: x})[0]
        y = np.asarray(y).reshape(-1).astype(np.float32)
        if y.shape[0] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {y.shape}")
        return y


class FaceEmbedder(Protocol):
    dim: int

    def embed_aligned_bgr112(self, face_bgr_112: np.ndarray) -> np.ndarray:  # (D,)
        ...


@dataclass(frozen=True)
class DeterministicFaceEmbedder:
    dim: int = 512

    def embed_aligned_bgr112(self, face_bgr_112: np.ndarray) -> np.ndarray:
        # Stable hash-like embedding from pixel statistics
        x = face_bgr_112.astype(np.float32)
        stats = np.array(
            [
                x.mean(),
                x.std(),
                x.min(),
                x.max(),
                np.median(x),
            ],
            dtype=np.float32,
        )
        v = np.tile(stats, int(np.ceil(self.dim / stats.shape[0])))[: self.dim]
        v = v / (np.linalg.norm(v) + 1e-12)
        return v.astype(np.float32)


@dataclass(frozen=True)
class InsightFaceArcFaceEmbedder:
    """
    Uses insightface to extract ArcFace embeddings.
    Expects already-aligned 112x112 BGR.
    """

    model_name: str = "buffalo_l"
    dim: int = 512

    def __post_init__(self) -> None:
        # Lazy-load on first call to avoid import cost for non-users.
        pass

    def _model(self):
        import insightface

        app = insightface.app.FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        return app

    def embed_aligned_bgr112(self, face_bgr_112: np.ndarray) -> np.ndarray:
        # insightface expects full image with detection; for aligned crops, use model_zoo.
        from insightface.model_zoo import get_model

        # Using a commonly-available ArcFace ONNX packaged with insightface for embeddings.
        model = get_model("arcface_r100_v1")
        model.prepare(ctx_id=-1)
        feat = model.get_feat(face_bgr_112)
        feat = np.asarray(feat).reshape(-1).astype(np.float32)
        if feat.shape[0] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {feat.shape}")
        return feat

