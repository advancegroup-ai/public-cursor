import numpy as np

from atlas_forgery.embedding import (
    DeterministicFaceEmbedder,
    DeterministicImageEmbedder,
    FaceMaskedDocFrontEmbedder,
)
from atlas_forgery.io import BBoxXYXY


def test_face_mask_changes_doc_vector() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[20:40, 20:40] = 100
    base = DeterministicImageEmbedder(output_dim=8)
    emb = FaceMaskedDocFrontEmbedder(base_embedder=base)
    v1 = emb.embed(img, None)
    v2 = emb.embed(img, BBoxXYXY(20, 20, 40, 40))
    assert v1.shape == (8,)
    assert v2.shape == (8,)
    assert not np.allclose(v1, v2)


def test_face_embedder_output_shape_and_norm() -> None:
    face = np.full((112, 112, 3), 128, dtype=np.uint8)
    emb = DeterministicFaceEmbedder(output_dim=512)
    v = emb.get_feature(face)
    assert v.shape == (512,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)
