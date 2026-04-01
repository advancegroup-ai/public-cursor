import numpy as np
 
from forgery_detection.cluster import cluster_by_cosine_threshold
from forgery_detection.masking import BBox, mask_bbox_zero_rgb
 
 
def test_mask_bbox_zero_rgb():
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    out = mask_bbox_zero_rgb(img, BBox(2, 3, 7, 8))
    assert out.shape == img.shape
    assert out[3:8, 2:7].sum() == 0
    assert out.sum() < img.sum()
    assert img.sum() > 0  # original unchanged
 
 
def test_cluster_connected_components():
    # v0 ~ v1, v2 isolated
    v0 = np.array([1.0, 0.0], dtype=np.float32)
    v1 = np.array([0.99, 0.01], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    vecs = np.stack([v0, v1, v2], axis=0)
    ids = ["a", "b", "c"]
    clusters = cluster_by_cosine_threshold(ids, vecs, threshold=0.95)
    sizes = sorted(len(c.member_indices) for c in clusters)
    assert sizes == [1, 2]
