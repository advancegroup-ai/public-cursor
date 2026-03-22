from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class OnnxSessionLike(Protocol):
    def get_inputs(self): ...
    def run(self, output_names, input_feed): ...


@dataclass(frozen=True)
class ClipPreprocessConfig:
    image_size: int = 224
    mean_rgb: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std_rgb: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


def _resize_bilinear_hwc(img_rgb: np.ndarray, out_size: int) -> np.ndarray:
    # Pure numpy bilinear resize to avoid hard cv2 dependency in core logic.
    # (CLI path uses cv2 if available; tests use this.)
    in_h, in_w = img_rgb.shape[:2]
    if in_h == out_size and in_w == out_size:
        return img_rgb

    ys = np.linspace(0, in_h - 1, out_size, dtype=np.float32)
    xs = np.linspace(0, in_w - 1, out_size, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    wx = (xs - x0).astype(np.float32)
    wy = (ys - y0).astype(np.float32)

    out = np.empty((out_size, out_size, 3), dtype=np.float32)
    for i, (yy0, yy1, wyy) in enumerate(zip(y0, y1, wy, strict=True)):
        top = (1.0 - wx)[:, None] * img_rgb[yy0, x0] + wx[:, None] * img_rgb[yy0, x1]
        bot = (1.0 - wx)[:, None] * img_rgb[yy1, x0] + wx[:, None] * img_rgb[yy1, x1]
        out[i] = (1.0 - wyy) * top + wyy * bot
    return out


def clip_image_to_input_tensor(
    img_bgr_u8: np.ndarray, *, cfg: ClipPreprocessConfig = ClipPreprocessConfig()
) -> np.ndarray:
    """
    Convert BGR uint8 HxWx3 image into CLIP float32 tensor [1,3,H,W] in RGB.
    """
    if img_bgr_u8.ndim != 3 or img_bgr_u8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape={img_bgr_u8.shape}")
    if img_bgr_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={img_bgr_u8.dtype}")

    img_rgb = img_bgr_u8[..., ::-1].astype(np.float32) / 255.0
    img_rgb = _resize_bilinear_hwc(img_rgb, cfg.image_size)
    mean = np.array(cfg.mean_rgb, dtype=np.float32)[None, None, :]
    std = np.array(cfg.std_rgb, dtype=np.float32)[None, None, :]
    img_rgb = (img_rgb - mean) / std
    chw = np.transpose(img_rgb, (2, 0, 1))
    return chw[None, ...].astype(np.float32, copy=False)


class ClipBackgroundEmbedder:
    """
    Thin wrapper around a CLIP image encoder ONNX session.

    Expected output: a single 512-d float vector (optionally batched).
    """

    def __init__(
        self,
        session: OnnxSessionLike,
        *,
        input_name: str | None = None,
        output_name: str | None = None,
        preprocess: ClipPreprocessConfig = ClipPreprocessConfig(),
        l2_normalize: bool = True,
    ) -> None:
        self._session = session
        self._preprocess = preprocess
        self._l2_normalize = l2_normalize
        if input_name is None:
            inputs = session.get_inputs()
            if not inputs:
                raise ValueError("ONNX session has no inputs")
            input_name = inputs[0].name
        self._input_name = input_name
        self._output_name = output_name

    def embed_bgr(self, img_bgr_u8: np.ndarray) -> np.ndarray:
        x = clip_image_to_input_tensor(img_bgr_u8, cfg=self._preprocess)
        outputs = self._session.run(
            None if self._output_name is None else [self._output_name],
            {self._input_name: x},
        )
        if not outputs:
            raise ValueError("ONNX session returned no outputs")
        vec = outputs[0]
        vec = np.asarray(vec)
        if vec.ndim == 2 and vec.shape[0] == 1:
            vec = vec[0]
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got shape={vec.shape}")
        vec = vec.astype(np.float32, copy=False)
        if self._l2_normalize:
            n = float(np.linalg.norm(vec) + 1e-12)
            vec = vec / n
        return vec

