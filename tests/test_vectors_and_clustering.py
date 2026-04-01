import numpy as np

from atlas_forgery.clustering import threshold_graph_clusters
from atlas_forgery.vectors import VectorStore, load_npz, save_npz


def test_vectorstore_roundtrip(tmp_path):
    ids = np.array(["a", "b", "c"], dtype=object)
    vecs = np.eye(3, dtype=np.float32)
    meta = {"source": "unit-test", "n": 3}
    store = VectorStore(ids=ids, vectors=vecs, meta=meta)
    p = tmp_path / "x.npz"
    save_npz(p, store)
    loaded = load_npz(p)
    assert loaded.ids.tolist() == ids.tolist()
    assert np.allclose(loaded.vectors, vecs)
    assert loaded.meta == meta


def test_threshold_graph_clusters_two_clusters():
    # Two tight groups in 2D, cosine threshold splits them.
    v = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.05],
            [0.0, 1.0],
            [0.05, 0.99],
        ],
        dtype=np.float32,
    )
    res = threshold_graph_clusters(v, threshold=0.9)
    assert res.n_clusters == 2
    # First two must share label; last two must share label.
    assert res.labels[0] == res.labels[1]
    assert res.labels[2] == res.labels[3]
    assert res.labels[0] != res.labels[2]

