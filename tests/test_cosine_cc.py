import numpy as np

from forgery_detection.clustering.cosine_cc import cluster_by_cosine_threshold


def test_cluster_two_blobs():
    a = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    b = np.array([[-1.0, 0.0], [-0.9, -0.1]], dtype=np.float32)
    x = np.concatenate([a, b], axis=0)
    res = cluster_by_cosine_threshold(x, threshold=0.8)
    assert res.n_clusters == 2
    assert len(set(res.labels.tolist())) == 2


def test_threshold_range_validation():
    x = np.random.randn(3, 4).astype(np.float32)
    try:
        cluster_by_cosine_threshold(x, threshold=1.5)
        assert False, "expected ValueError"
    except ValueError:
        assert True

