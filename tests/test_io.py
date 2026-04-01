import numpy as np

from atlas_forgery.io import load_npz, save_npz


def test_npz_roundtrip(tmp_path) -> None:
    ids = np.asarray(["a", "b", "c"])
    vecs = np.arange(15, dtype=np.float32).reshape(3, 5)
    meta = {"k": "v", "n": 3}

    p = tmp_path / "x.npz"
    save_npz(p, ids, vecs, meta=meta)
    store = load_npz(p)

    assert store.ids.tolist() == ["a", "b", "c"]
    assert store.vectors.shape == (3, 5)
    assert np.allclose(store.vectors, vecs)
    assert store.meta == meta

