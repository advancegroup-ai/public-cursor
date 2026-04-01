import numpy as np

from forgery_detection.clustering import cluster_by_cosine_threshold
from forgery_detection.image_ops import mask_bbox_zero_rgb
from forgery_detection.join import join_on_normalized_ids
from forgery_detection.npz_store import NpzVectorStore


def test_npz_store_roundtrip(tmp_path):
    path = tmp_path / "embeddings.npz"
    store = NpzVectorStore(
        ids=np.array(["A", "B"], dtype="U"),
        vectors=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    store.save(path)
    loaded = NpzVectorStore.load(path)
    assert loaded.ids.tolist() == ["A", "B"]
    np.testing.assert_allclose(loaded.vectors, store.vectors)


def test_join_on_normalized_ids():
    left = NpzVectorStore(
        ids=np.array([" Sig-1 ", "Sig-2", "SIG-3"], dtype="U"),
        vectors=np.eye(3, dtype=np.float32),
    )
    right = NpzVectorStore(
        ids=np.array(["sig-2", "sig-3", "sig-9"], dtype="U"),
        vectors=np.eye(3, dtype=np.float32),
    )

    lj, rj, report = join_on_normalized_ids(left, right)
    assert lj.ids.tolist() == ["Sig-2", "SIG-3"]
    assert rj.ids.tolist() == ["sig-2", "sig-3"]
    assert report.intersection_unique_norm == 2
    assert report.left_rows_joined == 2


def test_cluster_by_cosine_threshold():
    vectors = np.array(
        [
            [1.0, 0.0],  # cluster 1
            [0.99, 0.01],  # cluster 1
            [0.0, 1.0],  # cluster 2
        ],
        dtype=np.float32,
    )
    result = cluster_by_cosine_threshold(vectors, threshold=0.95)
    assert result.n_clusters == 2
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] != result.labels[0]


def test_mask_bbox_zero_rgb():
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    out = mask_bbox_zero_rgb(img, top_left=(2, 3), bottom_right=(5, 7))
    assert np.all(out[3:7, 2:5] == 0)
    assert np.all(out[0:2, 0:2] == 255)
