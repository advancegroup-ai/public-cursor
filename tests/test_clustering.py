import numpy as np

from forgery_detection.clustering.cosine_cc import cluster_by_cosine_threshold


def test_cosine_cc_clusters_expected() -> None:
    # Two tight groups (A,B) and (C,D)
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.99, 0.01], dtype=np.float32)
    c = np.array([0.0, 1.0], dtype=np.float32)
    d = np.array([0.01, 0.99], dtype=np.float32)
    x = np.stack([a, b, c, d], axis=0)
    res = cluster_by_cosine_threshold(x, threshold=0.95)
    assert res.n_clusters == 2
    labs = res.labels.tolist()
    assert labs[0] == labs[1]
    assert labs[2] == labs[3]
    assert labs[0] != labs[2]

