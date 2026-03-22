import numpy as np

from document_face_embeddings.clustering import threshold_clusters


def test_threshold_clusters_two_groups() -> None:
    # Two groups around orthogonal directions.
    vecs = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
            [0.02, 0.98, 0.0],
        ],
        dtype=np.float32,
    )
    res = threshold_clusters(vecs, threshold=0.95)
    assert res.n_clusters == 2
    labels = res.labels.tolist()
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]

