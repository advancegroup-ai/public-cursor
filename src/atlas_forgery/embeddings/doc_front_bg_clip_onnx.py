from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class MaskBox:
    x0: int
    y0: int
    x1: int
    y1: int


def _letterbox_rgb(img_rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _prepare_clip_input(img_bgr: np.ndarray, input_size: int = 224) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = _letterbox_rgb(img_rgb, input_size).astype(np.float32) / 255.0

    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img = (img - mean) / std

    # NCHW
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img


class DocFrontBackgroundEmbedderONNX:
    """
    ONNX CLIP-style background embedding with optional masking.

    This is intentionally minimal and does NOT include card boundary detection or face detection.
    Instead, callers can supply `mask_boxes` (e.g., face bbox on the card) and we will zero those pixels.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        providers: Optional[list[str]] = None,
    ) -> None:
        p = str(onnx_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            p,
            sess_options=sess_options,
            providers=providers or ort.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed_bgr(
        self,
        img_bgr: np.ndarray,
        mask_boxes: Optional[Iterable[MaskBox]] = None,
        input_size: int = 224,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        img = img_bgr.copy()
        if mask_boxes:
            h, w = img.shape[:2]
            for b in mask_boxes:
                x0 = max(0, min(w, b.x0))
                x1 = max(0, min(w, b.x1))
                y0 = max(0, min(h, b.y0))
                y1 = max(0, min(h, b.y1))
                if x1 > x0 and y1 > y0:
                    img[y0:y1, x0:x1] = 0

        inp = _prepare_clip_input(img, input_size=input_size)
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        vec = np.asarray(out).reshape(-1).astype(np.float32)
        if l2_normalize:
            n = float(np.linalg.norm(vec) + 1e-12)
            vec = vec / n
        return vec

