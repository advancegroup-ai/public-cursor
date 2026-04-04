from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


class _ArcFaceBackend(Protocol):
  def get(self, aligned_face_bgr: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Return 512-d embedding for a 112x112 aligned face (BGR or RGB depending on backend)."""
    ...


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
  denom = np.linalg.norm(x, axis=-1, keepdims=True)
  return x / np.maximum(denom, eps)


@dataclass(frozen=True)
class ArcFaceEmbedder:
  """
  ArcFace 112x112 aligned face -> 512-d embedding.

  This class is backend-pluggable so tests and deployments can swap implementations.
  """

  backend: _ArcFaceBackend
  l2_normalize: bool = True

  def embed_aligned_face(self, aligned_face: np.ndarray) -> np.ndarray:
    if aligned_face.ndim != 3 or aligned_face.shape[:2] != (112, 112):
      raise ValueError("expected aligned face image of shape (112, 112, C)")
    vec = np.asarray(self.backend.get(aligned_face)).reshape(-1).astype(np.float32)
    if vec.shape[0] != 512:
      raise ValueError(f"expected 512-d embedding, got shape {vec.shape}")
    if self.l2_normalize:
      vec = _l2_normalize(vec)
    return vec


def build_insightface_arcface_backend(
  *,
  provider: Optional[str] = None,
  ctx_id: int = -1,
):
  """
  Optional convenience builder that uses insightface (if installed).

  Note: This intentionally avoids importing heavy deps at module import time.
  """
  import insightface  # type: ignore

  # The most common drop-in is insightface.model_zoo.get_model("arcface_r100_v1")
  # but deployments may supply custom weights; keep this minimal.
  model = insightface.model_zoo.get_model(
    "arcface_r100_v1", providers=None if provider is None else [provider]
  )
  model.prepare(ctx_id=ctx_id)

  class _Backend:
    def get(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
      return model.get_feat(aligned_face_bgr).reshape(-1)

  return _Backend()
