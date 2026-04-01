import numpy as np

from atlas_forgery.cluster import threshold_graph_clusters


def test_threshold_graph_clusters_two_components() -> None:
    # Two tight pairs far apart in cosine space
    a1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    a2 = np.array([0.99, 0.01, 0.0], dtype=np.float32)
    b1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    b2 = np.array([0.01, 0.99, 0.0], dtype=np.float32)
    x = np.stack([a1, a2, b1, b2], axis=0)

    res = threshold_graph_clusters(x, threshold=0.95)
    clusters = [sorted(c) for c in res.clusters]
    clusters = sorted(clusters, key=lambda c: (len(c), c))
    assert clusters == [[0, 1], [2, 3]]

