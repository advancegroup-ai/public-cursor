import numpy as np

from forgery_detection.clustering.cosine_cc import cluster_by_cosine_threshold


def test_two_clusters_simple():
    # Two tight groups in 2D, already normalized
    a = np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)
    b = np.array([[0.0, 1.0], [0.01, 0.99]], dtype=np.float32)
    x = np.vstack([a, b])
    res = cluster_by_cosine_threshold(x, threshold=0.95)
    assert len(res.clusters) == 2
    assert sorted(map(len, res.clusters)) == [2, 2]


def test_empty_vectors():
    res = cluster_by_cosine_threshold(np.zeros((0, 512), dtype=np.float32), threshold=0.9)
    assert res.labels.shape == (0,)
    assert res.clusters == []

