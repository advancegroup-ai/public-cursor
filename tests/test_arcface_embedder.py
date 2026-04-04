import numpy as np
from PIL import Image

from forgery_detection.embeddings.arcface import ArcFaceEmbedder
from forgery_detection.embeddings.image_utils import BBox


class _FakeBackend:
    def embed_aligned_112(self, face_rgb_112: np.ndarray) -> np.ndarray:
        assert face_rgb_112.shape == (112, 112, 3)
        # Deterministic 512-dim embedding.
        return np.arange(512, dtype=np.float32)


def test_arcface_embedder_returns_512_and_l2_normalized():
    emb = ArcFaceEmbedder(backend=_FakeBackend())
    img = Image.new("RGB", (200, 100), (123, 234, 45))
    vec = emb.embed(img)
    assert vec.shape == (512,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_arcface_embedder_accepts_face_bbox_crop():
    emb = ArcFaceEmbedder(backend=_FakeBackend())
    img = Image.new("RGB", (300, 300), (0, 0, 0))
    vec = emb.embed(img, face_bbox=BBox(x1=50, y1=60, x2=250, y2=260))
    assert vec.shape == (512,)

