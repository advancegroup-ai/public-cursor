import numpy as np

from document_face_embeddings.vector_store import load_npz, save_npz


def test_save_load_npz_roundtrip(tmp_path) -> None:
    out = tmp_path / "vecs.npz"
    ids = ["a", "b", "c"]
    vecs = np.random.RandomState(0).randn(3, 5).astype(np.float32)
    save_npz(out, ids=ids, vectors=vecs)
    store = load_npz(out)
    assert store.ids.tolist() == ids
    assert store.vectors.shape == (3, 5)
    assert np.allclose(store.vectors, vecs)

