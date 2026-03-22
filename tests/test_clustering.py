import numpy as np

from atlas_forgery.clustering import threshold_graph_clusters


def test_threshold_graph_clusters_two_clusters():
    # Two tight clusters in 2D, normalized
    a = np.array([[1.0, 0.0], [0.99, 0.05]], dtype=np.float32)
    b = np.array([[0.0, 1.0], [0.05, 0.99]], dtype=np.float32)
    x = np.vstack([a, b])
    res = threshold_graph_clusters(x, threshold=0.9)
    assert res.labels.shape == (4,)
    assert len(res.sizes) == 2
    assert sorted(res.sizes.tolist()) == [2, 2]

