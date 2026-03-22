from pathlib import Path

import numpy as np

from atlas_forgery.vector_store import load_npz, save_npz


def test_save_load_npz_roundtrip(tmp_path: Path):
    ids = np.array(["a", "b"], dtype="<U8")
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = tmp_path / "v.npz"

    save_npz(out, ids, vecs)
    store = load_npz(out)

    assert store.ids.tolist() == ["a", "b"]
    assert store.vectors.shape == (2, 2)
    assert store.vectors.dtype == np.float32
