from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image


class OnnxSessionLike(Protocol):
    def run(self, output_names: Sequence[str] | None, input_feed: dict[str, Any]) -> list[Any]: ...


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(vec, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return vec / denom


def _clip_style_preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    """
    CLIP-ish preprocess: resize -> center crop -> RGB -> float -> normalize.

    Returns NCHW float32 with batch dim: (1, 3, size, size)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)

    left = (nw - size) // 2
    top = (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))

    arr = np.asarray(img).astype(np.float32) / 255.0
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))

    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std
    return arr[None, ...]


def _mask_face_bbox_rgb(img: Image.Image, bbox_xyxy: tuple[int, int, int, int]) -> Image.Image:
    """
    Zero out face pixels in-place (RGB).

    bbox format: (x1, y1, x2, y2) in pixel coords, inclusive-exclusive.
    """
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        return img

    arr = np.asarray(img.convert("RGB")).copy()
    h, w = arr.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return Image.fromarray(arr, mode="RGB")

    arr[y1c:y2c, x1c:x2c] = 0
    return Image.fromarray(arr, mode="RGB")


@dataclass(frozen=True)
class DocFrontBackgroundEmbedder:
    """
    Compute a 512-dim doc-front background embedding with optional face masking.

    This class assumes a card crop has already happened upstream. Face masking is done by
    zeroing pixels inside the provided bbox before CLIP-style preprocessing.
    """

    session: OnnxSessionLike
    input_name: str = "image"
    output_name: str | None = None
    l2_normalize: bool = True

    def embed(
        self,
        image: Image.Image,
        *,
        face_bbox_xyxy: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        if face_bbox_xyxy is not None:
            image = _mask_face_bbox_rgb(image, face_bbox_xyxy)

        inp = _clip_style_preprocess(image)
        out_list = self.session.run([self.output_name] if self.output_name else None, {self.input_name: inp})
        if not out_list:
            raise ValueError("ONNX session returned no outputs")

        vec = np.asarray(out_list[0]).astype(np.float32)
        vec = vec.reshape(-1)
        if vec.shape[0] != 512:
            raise ValueError(f"Expected 512-dim embedding, got shape {vec.shape}")

        if self.l2_normalize:
            vec = _l2_normalize(vec)
        return vec

