import numpy as np

from document_face_embeddings.vector_store import VectorStore, load_npz, save_npz


def test_vector_store_roundtrip(tmp_path) -> None:
    p = tmp_path / "emb.npz"
    ids = ["a", "b"]
    vec = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    save_npz(p, VectorStore(ids=ids, vectors=vec))
    out = load_npz(p)
    assert out.ids == ids
    assert out.vectors.dtype == np.float32
    assert out.vectors.shape == (2, 3)
    assert np.allclose(out.vectors, vec)

