import numpy as np
 
from forgery_detection.join import join_on_normalized_ids
from forgery_detection.vector_store import NpzVectorStore
 
 
def test_npz_store_roundtrip(tmp_path):
    store = NpzVectorStore(ids=["A", "B"], vectors=np.eye(2, dtype=np.float32))
    p = tmp_path / "x.npz"
    store.save(p)
    loaded = NpzVectorStore.load(p)
    assert loaded.ids == ["A", "B"]
    assert loaded.vectors.shape == (2, 2)
    assert loaded.vectors.dtype == np.float32
 
 
def test_join_normalizes_ids():
    left = NpzVectorStore(ids=[" SIG=ABC ", "def"], vectors=np.zeros((2, 3), dtype=np.float32))
    right = NpzVectorStore(ids=["abc", "DEF"], vectors=np.ones((2, 3), dtype=np.float32))
    joined = join_on_normalized_ids(left, right)
    assert joined.report.overlap_count == 2
    assert joined.left.ids == [" SIG=ABC ", "def"]
    assert joined.right.ids == ["abc", "DEF"]
