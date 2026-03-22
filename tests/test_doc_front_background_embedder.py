import numpy as np
from PIL import Image

from forgery_detection.embeddings.doc_front_background import DocFrontBackgroundEmbedder


class FakeSession:
    def __init__(self, out: np.ndarray):
        self.out = out
        self.last_feed = None

    def run(self, output_names, input_feed):
        self.last_feed = input_feed
        return [self.out]


def test_embedder_enforces_512d_and_normalizes():
    vec = np.ones((512,), dtype=np.float32)
    sess = FakeSession(vec.copy())
    emb = DocFrontBackgroundEmbedder(session=sess, l2_normalize=True)
    img = Image.new("RGB", (300, 200), color=(255, 0, 0))
    out = emb.embed(img)
    assert out.shape == (512,)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)
    assert "image" in sess.last_feed
    assert sess.last_feed["image"].shape == (1, 3, 224, 224)


def test_embedder_rejects_wrong_dim():
    vec = np.ones((256,), dtype=np.float32)
    sess = FakeSession(vec)
    emb = DocFrontBackgroundEmbedder(session=sess)
    img = Image.new("RGB", (224, 224))
    try:
        emb.embed(img)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "512" in str(e)


def test_face_bbox_masking_changes_input_tensor():
    vec = np.ones((512,), dtype=np.float32)
    sess1 = FakeSession(vec)
    sess2 = FakeSession(vec)
    emb1 = DocFrontBackgroundEmbedder(session=sess1, l2_normalize=False)
    emb2 = DocFrontBackgroundEmbedder(session=sess2, l2_normalize=False)

    # Image with a distinct green box in the center.
    img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    arr = np.array(img)
    arr[80:140, 80:140] = (0, 255, 0)
    img = Image.fromarray(arr, mode="RGB")

    emb1.embed(img, face_bbox_xyxy=None)
    emb2.embed(img, face_bbox_xyxy=(80, 80, 140, 140))

    t1 = sess1.last_feed["image"]
    t2 = sess2.last_feed["image"]
    # Masking should change the preprocessed tensor.
    assert np.mean(np.abs(t1 - t2)) > 0.0

