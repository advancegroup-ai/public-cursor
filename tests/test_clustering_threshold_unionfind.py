import numpy as np

from forgery_detection.clustering.threshold_unionfind import cluster_by_cosine_threshold


def test_cluster_by_cosine_threshold_connects_components():
    # v0 ~ v1 (high sim), v2 opposite, v3 unrelated
    v0 = np.array([1.0, 0.0], dtype=np.float32)
    v1 = np.array([0.99, 0.01], dtype=np.float32)
    v2 = np.array([-1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0], dtype=np.float32)
    vecs = np.stack([v0, v1, v2, v3], axis=0)
    labels = cluster_by_cosine_threshold(vecs, threshold=0.95)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    assert labels[3] != labels[0]


def test_empty_vectors():
    assert cluster_by_cosine_threshold(np.zeros((0, 512), dtype=np.float32), threshold=0.9) == []

