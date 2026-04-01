import numpy as np

from document_face_embeddings.embedding import MaskedEmbedder, MeanRGBEmbedder, mask_face_region
from document_face_embeddings.io import BBoxXYXY


def test_mask_face_region_zeroes_bbox() -> None:
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    bb = BBoxXYXY(x1=2, y1=3, x2=7, y2=6)
    masked = mask_face_region(img, face_bbox=bb)
    assert masked.shape == img.shape
    assert masked[3:6, 2:7].sum() == 0
    assert masked.sum() < img.sum()


def test_masked_embedder_changes_embedding_when_bbox_present() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, :] = [10, 20, 30]
    img[1:3, 1:3] = [250, 250, 250]

    base = MeanRGBEmbedder()
    emb = MaskedEmbedder(base=base)
    v_nomask = emb.embed(img, face_bbox=None)
    v_mask = emb.embed(img, face_bbox=BBoxXYXY(1, 1, 3, 3))
    assert v_nomask.shape == (3,)
    assert v_mask.shape == (3,)
    assert not np.allclose(v_nomask, v_mask)

