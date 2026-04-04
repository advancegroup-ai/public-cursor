import numpy as np
from PIL import Image

from forgery_detection.embeddings.doc_front_background import DocFrontBackgroundEmbedder


class _FakeSession:
    def __init__(self, out: np.ndarray):
        self._out = out

    def run(self, output_names, input_feed):
        # Validate input tensor exists and is NCHW
        x = next(iter(input_feed.values()))
        assert x.shape[0] == 1 and x.shape[1] == 3
        return [self._out]


def test_doc_front_background_embedder_enforces_512d_and_normalizes():
    raw = np.ones((1, 512), dtype=np.float32)
    sess = _FakeSession(raw)
    emb = DocFrontBackgroundEmbedder(session=sess, input_name="input", l2_normalize=True)
    img = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8), mode="RGB")
    v = emb.embed_pil(img)
    assert v.shape == (512,)
    # L2 norm should be ~1.0
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4


def test_doc_front_background_face_bbox_masking_does_not_crash():
    raw = np.arange(512, dtype=np.float32)[None, :]
    sess = _FakeSession(raw)
    emb = DocFrontBackgroundEmbedder(session=sess, input_name="input", l2_normalize=False)
    img = Image.fromarray(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8), mode="RGB")
    v = emb.embed_pil(img, face_bbox=(10, 20, 100, 120))
    assert v.shape == (512,)

