from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np
from PIL import Image


BBoxXYXY = Tuple[int, int, int, int]


class OnnxSessionLike(Protocol):
    def run(self, output_names, input_feed):  # pragma: no cover - Protocol
        ...


@dataclass(frozen=True)
class DocFrontBackgroundEmbedder:
    """
    Produces a 512-d embedding for a doc_front image, optionally masking out the face region.

    Notes:
    - Masking is a direct zero-fill on the RGB pixels inside bbox (x1,y1,x2,y2) in image coords.
    - Preprocess follows CLIP-style: resize -> center crop -> normalize.
    """

    session: OnnxSessionLike
    input_name: str = "input"
    output_name: Optional[str] = None
    image_size: int = 224
    normalize: bool = True
    l2_normalize: bool = True

    def embed(self, image: Image.Image, face_bbox_xyxy: Optional[BBoxXYXY] = None) -> np.ndarray:
        img = image.convert("RGB")
        if face_bbox_xyxy is not None:
            img = self._mask_face_bbox(img, face_bbox_xyxy)

        inp = self._preprocess(img)
        outputs = self.session.run(
            None if self.output_name is None else [self.output_name],
            {self.input_name: inp},
        )
        vec = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        if vec.shape[0] != 512:
            raise ValueError(f"Expected 512-d embedding, got shape {vec.shape}")
        if self.l2_normalize:
            vec = self._l2_normalize(vec)
        return vec

    @staticmethod
    def _mask_face_bbox(image: Image.Image, bbox_xyxy: BBoxXYXY) -> Image.Image:
        x1, y1, x2, y2 = bbox_xyxy
        arr = np.array(image, dtype=np.uint8)
        h, w = arr.shape[0], arr.shape[1]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return image
        arr[y1c:y2c, x1c:x2c] = 0
        return Image.fromarray(arr, mode="RGB")

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        # Resize shortest side to image_size, then center crop to image_size x image_size
        w, h = image.size
        scale = self.image_size / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = image.resize((new_w, new_h), resample=Image.BICUBIC)

        left = (new_w - self.image_size) // 2
        top = (new_h - self.image_size) // 2
        cropped = resized.crop((left, top, left + self.image_size, top + self.image_size))

        x = np.asarray(cropped, dtype=np.float32) / 255.0  # HWC RGB [0,1]
        x = np.transpose(x, (2, 0, 1))  # CHW

        if self.normalize:
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
            x = (x - mean) / std

        x = x[None, ...]  # NCHW
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        denom = float(np.linalg.norm(vec) + eps)
        return (vec / denom).astype(np.float32)

