from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence, Tuple

import numpy as np
from PIL import Image

BBoxXYXY = Tuple[int, int, int, int]


class OnnxLikeSession(Protocol):
    def run(self, output_names, input_feed): ...

    def get_inputs(self): ...


def _clip_preprocess_pil(
    img: Image.Image,
    *,
    size: int = 224,
    mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
    std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
) -> np.ndarray:
    """
    Convert PIL image to CLIP-normalized NCHW float32 tensor with shape (1,3,H,W).

    This mirrors the typical CLIP preprocessing:
    - RGB
    - Resize (square) using bicubic
    - Scale to [0,1]
    - Normalize with CLIP mean/std
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((size, size), resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, RGB
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr[None, ...]  # NCHW


def mask_face_pixels(img_rgb: np.ndarray, face_bbox_xyxy: BBoxXYXY) -> np.ndarray:
    """
    Zero-out the face area within an RGB image (H,W,3) using an xyxy bbox.

    - bbox is (x1, y1, x2, y2) in pixel coordinates
    - bbox is clipped to image bounds
    - returns a copy (does not modify input array)
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got shape={img_rgb.shape}")

    h, w, _ = img_rgb.shape
    x1, y1, x2, y2 = face_bbox_xyxy
    x1 = int(max(0, min(w, x1)))
    x2 = int(max(0, min(w, x2)))
    y1 = int(max(0, min(h, y1)))
    y2 = int(max(0, min(h, y2)))
    if x2 <= x1 or y2 <= y1:
        return img_rgb.copy()

    out = img_rgb.copy()
    out[y1:y2, x1:x2, :] = 0
    return out


def _infer_onnx_input_name(session: OnnxLikeSession) -> str:
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("ONNX session has no inputs")
    return inputs[0].name


@dataclass(frozen=True)
class DocFrontBackgroundEmbedder:
    """
    Doc-front background embedding (face excluded) using a CLIP-like ONNX model.

    The expected model output is a single 512-dim vector.
    """

    session: OnnxLikeSession
    input_size: int = 224
    l2_normalize: bool = True

    def embed_pil(
        self, img: Image.Image, *, face_bbox_xyxy: Optional[BBoxXYXY] = None
    ) -> np.ndarray:
        if face_bbox_xyxy is not None:
            rgb = np.asarray(img.convert("RGB"))
            rgb = mask_face_pixels(rgb, face_bbox_xyxy)
            img = Image.fromarray(rgb, mode="RGB")

        x = _clip_preprocess_pil(img, size=self.input_size)
        input_name = _infer_onnx_input_name(self.session)
        outputs = self.session.run(None, {input_name: x})
        if not outputs:
            raise ValueError("ONNX session returned no outputs")

        emb = np.asarray(outputs[0]).astype(np.float32).reshape(-1)
        if emb.shape[0] != 512:
            raise ValueError(f"Expected 512-dim embedding, got shape={emb.shape}")

        if self.l2_normalize:
            n = float(np.linalg.norm(emb) + 1e-12)
            emb = emb / n
        return emb


def load_onnx_session(model_path: str, *, providers: Optional[Iterable[str]] = None):
    """
    Load an ONNX Runtime session.

    Kept in a separate function to avoid importing onnxruntime unless needed.
    """
    import onnxruntime as ort  # type: ignore

    sess_opts = ort.SessionOptions()
    if providers is None:
        return ort.InferenceSession(model_path, sess_options=sess_opts)
    return ort.InferenceSession(
        model_path, sess_options=sess_opts, providers=list(providers)
    )
