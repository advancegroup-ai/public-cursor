import numpy as np

from atlas_forgery.vector_store import load_npz, save_npz


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "x.npz"
    ids = ["a", "b", "c"]
    vectors = np.eye(3, dtype=np.float32)
    save_npz(p, ids=ids, vectors=vectors, foo="bar", n=3)
    store = load_npz(p)
    assert store.ids.tolist() == ids
    assert store.vectors.shape == (3, 3)
    assert store.meta["foo"] == "bar"
    assert store.meta["n"] == 3

