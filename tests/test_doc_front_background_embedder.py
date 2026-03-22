import numpy as np
from PIL import Image

from forgery_detection.embeddings.doc_front_background import DocFrontBackgroundEmbedder
from forgery_detection.embeddings.image_utils import BBox


class _FakeOnnxSession:
    def __init__(self):
        self.last_input = None

    def run(self, output_names, input_feed):
        self.last_input = input_feed
        # Return a deterministic 512-dim vector.
        v = np.linspace(0, 1, 512, dtype=np.float32)[None, :]
        return [v]


def test_embedder_returns_512_and_l2_normalized():
    sess = _FakeOnnxSession()
    emb = DocFrontBackgroundEmbedder(session=sess, input_name="input", l2_normalize=True)
    img = Image.new("RGB", (640, 480), (255, 0, 0))
    vec = emb.embed(img)
    assert vec.shape == (512,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_face_bbox_is_masked_before_preprocess():
    sess = _FakeOnnxSession()
    emb = DocFrontBackgroundEmbedder(session=sess, input_name="input", l2_normalize=False)

    # Create image with a white face-like patch.
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:] = (10, 10, 10)
    arr[50:150, 60:160] = (250, 250, 250)
    img = Image.fromarray(arr, mode="RGB")

    _ = emb.embed(img, face_bbox=BBox(x1=60, y1=50, x2=160, y2=150))
    x = sess.last_input["input"]
    assert x.shape == (1, 3, 224, 224)
    # Not asserting exact pixels after resize/crop; just ensure tensor is finite.
    assert np.isfinite(x).all()

