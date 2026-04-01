import numpy as np

from forgery_detection.clustering import connected_components_cosine
from forgery_detection.image_masking import mask_bbox_zero_rgb


def test_mask_bbox_zero_rgb():
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    out = mask_bbox_zero_rgb(img, (2, 3, 6, 8))
    assert out[3:8, 2:6, :3].sum() == 0
    assert out[0, 0, 0] == 255
    assert img[3, 2, 0] == 255


def test_connected_components_cosine():
    v = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    res = connected_components_cosine(v, threshold=0.8)
    assert res.n_clusters == 2
    assert res.labels.tolist()[0] == res.labels.tolist()[1]
    assert res.labels.tolist()[2] != res.labels.tolist()[0]
