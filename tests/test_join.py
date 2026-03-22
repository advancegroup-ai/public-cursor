import numpy as np

from forgery_detection.id_normalize import IdNormalization
from forgery_detection.join import join_on_normalized_ids
from forgery_detection.npz_store import NpzVectorStore


def test_join_on_normalized_ids_basic_overlap() -> None:
    left = NpzVectorStore(ids=np.array([" A ", "B", "C"]), vectors=np.eye(3, dtype=np.float32))
    right = NpzVectorStore(ids=np.array(["a", "b", "D"]), vectors=np.eye(3, dtype=np.float32))

    norm = IdNormalization(trim=True, collapse_ws=True, lower=True)
    l2, r2, report = join_on_normalized_ids(left, right, norm=norm)

    assert report.overlap_n == 2
    assert l2.size == 2
    assert r2.size == 2
    assert set(l2.ids.tolist()) == {"a", "b"}
    assert np.allclose(l2.vectors, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    assert np.allclose(r2.vectors, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))


def test_join_on_normalized_ids_no_overlap() -> None:
    left = NpzVectorStore(ids=np.array(["x"]), vectors=np.zeros((1, 2), dtype=np.float32))
    right = NpzVectorStore(ids=np.array(["y"]), vectors=np.zeros((1, 2), dtype=np.float32))
    l2, r2, report = join_on_normalized_ids(left, right)
    assert report.overlap_n == 0
    assert l2.size == 0
    assert r2.size == 0

