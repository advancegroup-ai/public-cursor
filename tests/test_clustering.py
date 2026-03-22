import numpy as np

from forgery_detection.clustering.cosine_cc import cluster_by_cosine_threshold


def test_cluster_by_cosine_threshold_connected_components():
    # Two tight clusters: {0,1} and {2,3}
    v0 = np.array([1.0, 0.0], dtype=np.float32)
    v1 = np.array([0.99, 0.01], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    v3 = np.array([0.01, 0.99], dtype=np.float32)
    res = cluster_by_cosine_threshold([v0, v1, v2, v3], threshold=0.9)
    assert res.num_clusters == 2
    clusters = [sorted(c) for c in res.clusters()]
    assert sorted(clusters) == [[0, 1], [2, 3]]

