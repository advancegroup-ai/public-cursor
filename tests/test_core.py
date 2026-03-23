import numpy as np

from forgery_document_face_embeddings.core import (
    cluster_by_cosine_threshold,
    id_normalize,
    join_on_normalized_ids,
    mask_bbox_zero_rgb,
)


def test_id_normalize():
    assert id_normalize("  Sig-123_A  ") == "sig123a"
    assert id_normalize("") == ""


def test_join_report_intersection():
    left = ["SIG-001", " sig_002 ", "x"]
    right = ["sig001", "sig-003", "x"]
    rep = join_on_normalized_ids(left, right)
    assert rep.left_count == 3
    assert rep.right_count == 3
    assert rep.intersection == 2


def test_mask_bbox_zero_rgb():
    img = np.ones((5, 6, 3), dtype=np.uint8) * 255
    out = mask_bbox_zero_rgb(img, (1, 1, 4, 4))
    assert out.shape == img.shape
    assert np.all(out[1:4, 1:4, :] == 0)
    assert np.all(out[0, :, :] == 255)
    # original unchanged
    assert np.all(img == 255)


def test_cluster_by_cosine_threshold():
    # first two vectors close, third opposite-ish
    x = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.1],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    clusters = cluster_by_cosine_threshold(x, threshold=0.95)
    sizes = sorted([len(c) for c in clusters], reverse=True)
    assert sizes == [2, 1]
