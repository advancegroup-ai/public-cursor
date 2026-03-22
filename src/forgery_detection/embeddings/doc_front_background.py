from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from PIL import Image

from .image_utils import BBox, load_rgb_image, pil_to_rgb_numpy, zero_out_bbox


class OnnxLikeSession(Protocol):
    def run(self, output_names, input_feed):  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class ClipPreprocessConfig:
    size: int = 224
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


class DocFrontBackgroundEmbedder:
    """
    Compute a 512-dim CLIP-style embedding from a doc_front image, optionally
    masking the face region by zeroing its pixels.
    """

    def __init__(
        self,
        session: OnnxLikeSession,
        input_name: str = "input",
        output_name: Optional[str] = None,
        preprocess: ClipPreprocessConfig = ClipPreprocessConfig(),
        l2_normalize: bool = True,
    ) -> None:
        self._session = session
        self._input_name = input_name
        self._output_name = output_name
        self._pp = preprocess
        self._l2 = l2_normalize

    def embed(self, image: str | Image.Image, face_bbox: Optional[BBox] = None) -> np.ndarray:
        pil = load_rgb_image(image)
        rgb = pil_to_rgb_numpy(pil)
        rgb_masked = zero_out_bbox(rgb, face_bbox)
        x = self._preprocess(rgb_masked)

        output_names = None if self._output_name is None else [self._output_name]
        y = self._session.run(output_names, {self._input_name: x})[0]
        vec = np.asarray(y).reshape(-1).astype(np.float32)
        if vec.shape[0] != 512:
            raise ValueError(f"Expected 512-dim embedding, got shape={vec.shape}")
        if self._l2:
            n = float(np.linalg.norm(vec) + 1e-12)
            vec = vec / n
        return vec

    def _preprocess(self, img_rgb: np.ndarray) -> np.ndarray:
        # Resize shortest side and center-crop to size x size (CLIP-ish)
        pil = Image.fromarray(img_rgb)
        size = self._pp.size
        w, h = pil.size
        scale = size / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        pil = pil.resize((new_w, new_h), resample=Image.BICUBIC)

        left = (new_w - size) // 2
        top = (new_h - size) // 2
        pil = pil.crop((left, top, left + size, top + size))

        arr = np.asarray(pil).astype(np.float32) / 255.0  # HWC, RGB
        mean = np.array(self._pp.mean, dtype=np.float32)
        std = np.array(self._pp.std, dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        arr = np.expand_dims(arr, axis=0)  # NCHW
        return arr.astype(np.float32)

