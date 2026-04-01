import numpy as np

from atlas_forgery.cluster import threshold_graph_clusters


def test_threshold_graph_clusters_connected_components() -> None:
    # Two tight groups in 2D, normalized.
    a = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    b = np.array([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    x = np.vstack([a, b])

    clusters = threshold_graph_clusters(x, threshold=0.95)
    # Expect two clusters: first 3 together, last 2 together (order sorted by size desc)
    assert clusters[0] == [0, 1, 2]
    assert clusters[1] == [3, 4]

