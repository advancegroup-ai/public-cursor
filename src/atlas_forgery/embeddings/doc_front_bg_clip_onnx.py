from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


BBoxXYXY = Tuple[int, int, int, int]


def _to_rgb_pil(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _mask_bbox_black(rgb: np.ndarray, bbox_xyxy: BBoxXYXY) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(max(0, min(x1, rgb.shape[1])))
    x2 = int(max(0, min(x2, rgb.shape[1])))
    y1 = int(max(0, min(y1, rgb.shape[0])))
    y2 = int(max(0, min(y2, rgb.shape[0])))
    if x2 <= x1 or y2 <= y1:
        return rgb
    rgb[y1:y2, x1:x2, :] = 0
    return rgb


@dataclass(frozen=True)
class ClipPreprocess:
    size: int = 224
    mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    def __call__(self, img: Image.Image) -> np.ndarray:
        img = _to_rgb_pil(img).resize((self.size, self.size), Image.BICUBIC)
        arr = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW
        mean = np.asarray(self.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.std, dtype=np.float32)[:, None, None]
        arr = (arr - mean) / std
        return arr[None, ...]  # NCHW


class DocFrontBackgroundClipOnnxEmbedder:
    """
    Background embedding for doc_front images using a CLIP-style ONNX model.

    The intended workflow is:
    - optionally mask face region on the document (bbox in original image coords)
    - run ONNX encoder to obtain a 512-dim embedding
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[Iterable[str]] = None,
        preprocess: Optional[ClipPreprocess] = None,
        output_l2_normalize: bool = True,
    ) -> None:
        self.onnx_path = onnx_path
        self.preprocess = preprocess or ClipPreprocess()
        self.output_l2_normalize = output_l2_normalize
        sess_opts = ort.SessionOptions()
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=list(providers) if providers else None,
        )

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected 1 input, got {len(inputs)}: {[i.name for i in inputs]}")
        self.input_name = inputs[0].name

        outputs = self.session.get_outputs()
        if len(outputs) < 1:
            raise ValueError("ONNX model has no outputs")
        self.output_name = outputs[0].name

    def embed_pil(self, img: Image.Image, face_bbox_xyxy: Optional[BBoxXYXY] = None) -> np.ndarray:
        img = _to_rgb_pil(img)
        if face_bbox_xyxy is not None:
            rgb = np.asarray(img).copy()
            rgb = _mask_bbox_black(rgb, face_bbox_xyxy)
            img = Image.fromarray(rgb, mode="RGB")

        x = self.preprocess(img)
        y = self.session.run([self.output_name], {self.input_name: x})[0]
        y = np.asarray(y).reshape(-1).astype(np.float32)
        if self.output_l2_normalize:
            n = float(np.linalg.norm(y) + 1e-12)
            y = y / n
        return y

