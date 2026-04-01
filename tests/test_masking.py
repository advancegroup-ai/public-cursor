import numpy as np

from atlas_forgery.masking import BBoxXYXY, mask_bbox_zero_rgb


def test_mask_bbox_zero_rgb_basic():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    bbox = BBoxXYXY(2, 3, 5, 7)
    out = mask_bbox_zero_rgb(img, bbox)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert (out[3:7, 2:5] == 0).all()
    assert (out[:3, :] == 255).all()


def test_mask_bbox_zero_rgb_clips():
    img = np.full((4, 4, 3), 10, dtype=np.uint8)
    bbox = BBoxXYXY(-10, -10, 2, 2)
    out = mask_bbox_zero_rgb(img, bbox)
    assert (out[0:2, 0:2] == 0).all()
    assert (out[2:, 2:] == 10).all()

