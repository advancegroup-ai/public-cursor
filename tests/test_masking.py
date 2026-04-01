import numpy as np

from atlas_forgery.masking import mask_bbox_zero_rgb
from atlas_forgery.types import BBoxXYXY


def test_mask_bbox_zero_rgb_basic() -> None:
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    out = mask_bbox_zero_rgb(img, BBoxXYXY(2, 3, 5, 7))
    assert out.shape == img.shape
    assert (out[3:7, 2:5] == 0).all()
    assert (out[:3] == 255).any()


def test_mask_bbox_zero_rgb_clip_outside() -> None:
    img = np.ones((4, 4, 3), dtype=np.uint8) * 7
    out = mask_bbox_zero_rgb(img, BBoxXYXY(-10, -10, 2, 2))
    assert (out[0:2, 0:2] == 0).all()
    assert (out[2:, 2:] == 7).all()

