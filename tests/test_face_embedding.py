import numpy as np

from atlas_forgery.face_embedding import DeterministicFaceEmbedder, l2_normalize


def test_l2_normalize_unit_norm():
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = l2_normalize(v)
    assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-6)


def test_deterministic_face_embedder_dim_and_determinism():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[..., 0] = 10
    img[..., 1] = 20
    img[..., 2] = 30

    e = DeterministicFaceEmbedder(dim=512)
    v1 = e.embed(img)
    v2 = e.embed(img.copy())

    assert v1.shape == (512,)
    assert v1.dtype == np.float32
    assert np.allclose(v1, v2)
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)
