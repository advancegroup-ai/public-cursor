import numpy as np

from document_face_embeddings.clustering import (
    cluster_size_stats,
    cosine_similarity_matrix,
    threshold_connected_components,
)


def test_threshold_connected_components_two_clusters() -> None:
    # Two tight groups in 2D
    x = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float32,
    )
    sim = cosine_similarity_matrix(x)
    res = threshold_connected_components(sim, threshold=0.8)
    assert res.n_clusters == 2
    stats = cluster_size_stats(res.labels)
    assert stats["n_items"] == 4
    assert stats["max_cluster_size"] == 2

