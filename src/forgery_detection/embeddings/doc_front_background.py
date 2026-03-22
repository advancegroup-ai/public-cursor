from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np
from PIL import Image


class _OnnxLikeSession(Protocol):
  def run(self, output_names, input_feed):  # pragma: no cover
    ...


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
  denom = np.linalg.norm(x, axis=-1, keepdims=True)
  return x / np.maximum(denom, eps)


def _clip_preprocess_rgb_uint8(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
  """
  CLIP ViT-B/32 style preprocess: resize -> center crop -> normalize.
  Produces float32 tensor (1, 3, 224, 224).
  """
  if img_rgb.dtype != np.uint8 or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
    raise ValueError("expected RGB uint8 image of shape (H, W, 3)")

  pil = Image.fromarray(img_rgb, mode="RGB")
  # CLIP uses bicubic resize to 224 with center crop when aspect differs.
  pil = pil.resize((size, size), resample=Image.BICUBIC)
  arr = np.asarray(pil).astype(np.float32) / 255.0
  # normalize
  mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
  std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
  arr = (arr - mean) / std
  # HWC -> CHW
  chw = np.transpose(arr, (2, 0, 1))
  return chw[None, ...].astype(np.float32, copy=False)


def _mask_bbox_inplace(img_rgb: np.ndarray, bbox_xyxy: Sequence[int]) -> None:
  x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
  h, w = img_rgb.shape[:2]
  x1 = max(0, min(w, x1))
  x2 = max(0, min(w, x2))
  y1 = max(0, min(h, y1))
  y2 = max(0, min(h, y2))
  if x2 <= x1 or y2 <= y1:
    return
  img_rgb[y1:y2, x1:x2] = 0


@dataclass(frozen=True)
class DocFrontBackgroundEmbedder:
  """
  Compute doc_front background embedding with optional face exclusion (bbox masking).

  The intended ONNX model is a CLIP-based background embedding model producing a 512-d vector.
  """

  session: _OnnxLikeSession
  input_name: str = "input"
  output_name: Optional[str] = None
  l2_normalize: bool = True

  def embed_rgb_uint8(
    self, img_rgb: np.ndarray, *, face_bbox_xyxy: Optional[Sequence[int]] = None
  ) -> np.ndarray:
    if face_bbox_xyxy is not None:
      img_rgb = np.array(img_rgb, copy=True)
      _mask_bbox_inplace(img_rgb, face_bbox_xyxy)

    x = _clip_preprocess_rgb_uint8(img_rgb, size=224)
    out = self.session.run(
      None if self.output_name is None else [self.output_name],
      {self.input_name: x},
    )
    if not out:
      raise RuntimeError("ONNX session returned no outputs")
    vec = np.asarray(out[0]).reshape(-1).astype(np.float32)
    if vec.shape[0] != 512:
      raise ValueError(f"expected 512-d embedding, got shape {vec.shape}")
    if self.l2_normalize:
      vec = _l2_normalize(vec)
    return vec
 
