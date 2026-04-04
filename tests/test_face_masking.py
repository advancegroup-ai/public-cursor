import numpy as np

from forgery_detection.embeddings.face_masking import zero_out_bbox
from forgery_detection.types import BBoxXYXY


def test_zero_out_bbox_clips_and_zeros() -> None:
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    out = zero_out_bbox(img, BBoxXYXY(x1=-3, y1=2, x2=4, y2=20))
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out[2:10, 0:4].sum() == 0
    assert out[0:2, :, :].sum() > 0

