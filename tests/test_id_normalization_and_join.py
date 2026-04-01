import numpy as np

from forgery_detection.id_normalization import IdNormalizer
from forgery_detection.join import join_on_normalized_ids
from forgery_detection.npz_store import NpzVectorStore


def test_id_normalizer_basic():
    n = IdNormalizer()
    assert n.normalize(" SIG:ABC  ") == "abc"
    assert n.normalize("signature:  XYZ") == "xyz"
    assert n.normalize("uid:   A   B") == "a b"


def test_join_on_normalized_ids_aligns_vectors():
    left = NpzVectorStore(
        ids=np.array(["sig:1", "sig:2", "sig:3"]),
        vectors=np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
    )
    right = NpzVectorStore(
        ids=np.array(["1", "3", "4"]),
        vectors=np.array([[10, 0], [30, 0], [40, 0]], dtype=np.float32),
    )

    l2, r2, report = join_on_normalized_ids(left, right)
    assert report.overlap_norm == 2
    assert l2.ids.tolist() == ["sig:1", "sig:3"]
    assert r2.ids.tolist() == ["1", "3"]
    assert l2.vectors.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert r2.vectors.tolist() == [[10.0, 0.0], [30.0, 0.0]]
