import numpy as np

from atlas_forgery.cluster import threshold_graph_clusters


def test_threshold_graph_clusters_two_components() -> None:
    # Three points: two identical, one orthogonal-ish
    v = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    res = threshold_graph_clusters(v, threshold=0.99)
    sizes = sorted(len(c) for c in res.clusters)
    assert sizes == [1, 2]

