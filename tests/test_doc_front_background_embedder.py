import numpy as np
from PIL import Image

from forgery_detection.embeddings.doc_front_background import DocFrontBackgroundEmbedder


class FakeSession:
    def __init__(self, vec: np.ndarray):
        self.vec = vec.astype(np.float32).reshape(1, -1)
        self.last_input = None

    def run(self, output_names, input_feed):
        self.last_input = input_feed
        return [self.vec]


def test_masks_face_bbox_and_runs_session():
    img = Image.new("RGB", (100, 60), color=(255, 255, 255))
    vec = np.ones((512,), dtype=np.float32)
    sess = FakeSession(vec)
    emb = DocFrontBackgroundEmbedder(session=sess, input_name="input", l2_normalize=False)

    out = emb.embed(img, face_bbox_xyxy=(10, 10, 50, 40))
    assert out.shape == (512,)
    assert np.allclose(out, 1.0)
    assert "input" in sess.last_input
    assert sess.last_input["input"].shape == (1, 3, 224, 224)


def test_enforces_512_dim():
    img = Image.new("RGB", (50, 50), color=(0, 0, 0))
    sess = FakeSession(np.ones((128,), dtype=np.float32))
    emb = DocFrontBackgroundEmbedder(session=sess)
    try:
        emb.embed(img)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "512" in str(e)

