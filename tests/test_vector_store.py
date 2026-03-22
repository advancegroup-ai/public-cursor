import numpy as np

from atlas_forgery.vector_store import VectorStore, load_npz, save_npz


def test_vector_store_roundtrip(tmp_path) -> None:
    store = VectorStore(
        ids=["a", "b"],
        vectors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        meta={"kind": "test", "n": 2},
    )
    p = tmp_path / "vecs.npz"
    save_npz(p, store)
    loaded = load_npz(p)
    assert loaded.ids == store.ids
    assert loaded.vectors.shape == store.vectors.shape
    assert np.allclose(loaded.vectors, store.vectors)
    assert loaded.meta == store.meta

