import numpy as np

from atlas_forgery.masking import BBoxXYXY, mask_bbox_zero_rgb


def test_mask_bbox_zero_rgb_basic():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    out = mask_bbox_zero_rgb(img, BBoxXYXY(2, 3, 5, 7))
    assert out.shape == img.shape
    assert (out[3:7, 2:5] == 0).all()
    # outside remains 255
    assert (out[0:3, 0:2] == 255).all()


def test_mask_bbox_zero_rgb_clamps_and_noop():
    img = np.full((4, 4, 3), 10, dtype=np.uint8)
    out = mask_bbox_zero_rgb(img, BBoxXYXY(-10, -10, -1, -1))
    assert (out == img).all()

