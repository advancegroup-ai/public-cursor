import numpy as np

from atlas_forgery.npz_store import NpzVectorStore


def test_npz_roundtrip(tmp_path) -> None:
    ids = ["a", "b", "c"]
    vec = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    store = NpzVectorStore(ids=ids, vectors=vec)

    p = tmp_path / "x.npz"
    store.save(p)
    loaded = NpzVectorStore.load(p)

    assert loaded.ids == ids
    assert loaded.vectors.shape == (3, 4)
    assert np.allclose(loaded.vectors, vec)

