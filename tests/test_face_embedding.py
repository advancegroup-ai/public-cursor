import numpy as np

from document_face_embeddings.face_embedding import DeterministicFaceEmbedder


def test_deterministic_face_embedder_shape_and_norm() -> None:
    emb = DeterministicFaceEmbedder()
    face = np.full((112, 112, 3), 128, dtype=np.uint8)
    v = emb.embed_face_bgr(face)
    assert v.shape == (3,)
    assert np.isfinite(v).all()
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

