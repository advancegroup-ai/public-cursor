import numpy as np

from atlas_forgery.mask import BBoxXYXY, mask_bbox_zero_rgb


def test_mask_bbox_zero_rgb_clips_and_zeros() -> None:
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    bbox = BBoxXYXY(-5, 2, 6, 20)  # should clip to x:0..6, y:2..10
    out = mask_bbox_zero_rgb(img, bbox)
    assert out.shape == img.shape
    # region should be zero
    assert (out[2:10, 0:6] == 0).all()
    # outside region should remain 255
    assert (out[0:2, :, :] == 255).all()
    assert (out[:, 6:, :] == 255).all()

