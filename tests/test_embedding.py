import numpy as np

from document_face_embeddings.embedding import MaskedEmbedder, MeanRGBEmbedder
from document_face_embeddings.io import BBoxXYXY


def test_face_mask_changes_embedding() -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :, :] = [10, 20, 30]
    img[2:6, 2:6, :] = [250, 250, 250]

    emb = MaskedEmbedder(base=MeanRGBEmbedder(normalize=False))
    v_raw = emb.embed(img, face_bbox=None)
    v_mask = emb.embed(img, face_bbox=BBoxXYXY(2, 2, 6, 6))

    assert v_raw.shape == (3,)
    assert v_mask.shape == (3,)
    assert not np.allclose(v_raw, v_mask)
    assert float(v_mask.mean()) < float(v_raw.mean())

