import numpy as np

from atlas_forgery.doc_embed import mask_bbox_zero_rgb


def test_mask_bbox_zero_rgb_zeros_region() -> None:
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    out = mask_bbox_zero_rgb(img, (2, 3, 7, 8))
    assert out.shape == img.shape
    assert int(out[3:8, 2:7].sum()) == 0
    assert int(out.sum()) < int(img.sum())


def test_mask_bbox_zero_rgb_clips_bounds() -> None:
    img = np.ones((5, 5, 3), dtype=np.uint8) * 10
    out = mask_bbox_zero_rgb(img, (-100, -100, 100, 100))
    assert int(out.sum()) == 0

