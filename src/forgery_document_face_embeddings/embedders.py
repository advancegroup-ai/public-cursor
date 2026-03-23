from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .core import mask_bbox_zero_rgb


def _require_ort():
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "onnxruntime is required for BackgroundEmbedderONNX; install extra 'onnx'"
        ) from e
    return ort


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("opencv-python is required; install extra 'cv'") from e
    return cv2


@dataclass
class BackgroundEmbedderONNX:
    """
    CLIP-like background embedding with optional face bbox masking.

    Notes:
    - Expects model output to be one vector per image.
    - Input preprocessing is generic CLIP-style normalization.
    """

    onnx_path: str
    input_size: int = 224

    def __post_init__(self) -> None:
        ort = _require_ort()
        self._cv2 = _require_cv2()
        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x.astype(np.float32)

    def embed(
        self,
        img_bgr: np.ndarray,
        face_bbox_xyxy: Sequence[int | float] | None = None,
    ) -> np.ndarray:
        """
        Return a 512-d (or model output dim) normalized vector.
        """
        work = img_bgr
        if face_bbox_xyxy is not None:
            work = mask_bbox_zero_rgb(work, face_bbox_xyxy)
        inp = self._preprocess(work)
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        vec = np.asarray(out).reshape(-1).astype(np.float32)
        n = float(np.linalg.norm(vec) + 1e-12)
        return vec / n


@dataclass
class ArcFaceEmbedder:
    """
    ArcFace face embedding wrapper using InsightFace.

    Produces normalized embedding vectors (typically 512-dim).
    """

    det_size: tuple[int, int] = (640, 640)
    allowed_modules: tuple[str, ...] = ("detection", "recognition")

    def __post_init__(self) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "insightface is required for ArcFaceEmbedder; install extra 'face'"
            ) from e

        self.app = FaceAnalysis(allowed_modules=list(self.allowed_modules))
        self.app.prepare(ctx_id=-1, det_size=self.det_size)

    def embed_largest_face(self, img_bgr: np.ndarray) -> np.ndarray:
        faces = self.app.get(img_bgr)
        if not faces:
            raise ValueError("No face detected")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        vec = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(vec) + 1e-12)
        return vec / n
