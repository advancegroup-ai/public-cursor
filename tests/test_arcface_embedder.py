import numpy as np
import pytest

from forgery_detection.embeddings.arcface import ArcFaceEmbedder


class FakeBackend:
  def __init__(self, vec: np.ndarray):
    self.vec = vec

  def get(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
    return self.vec


def test_arcface_embedder_enforces_shape_and_norm():
  backend = FakeBackend(np.ones(512, dtype=np.float32))
  emb = ArcFaceEmbedder(backend=backend)
  face = np.zeros((112, 112, 3), dtype=np.uint8)
  v = emb.embed_aligned_face(face)
  assert v.shape == (512,)
  assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)


def test_arcface_embedder_rejects_wrong_image_shape():
  backend = FakeBackend(np.ones(512, dtype=np.float32))
  emb = ArcFaceEmbedder(backend=backend)
  with pytest.raises(ValueError):
    emb.embed_aligned_face(np.zeros((111, 112, 3), dtype=np.uint8))
