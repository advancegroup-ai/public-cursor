import numpy as np
from PIL import Image
 
from forgery_detection.embeddings.doc_front_background import (
    DocFrontBackgroundEmbedder,
    mask_face_pixels,
)
 
 
def test_mask_face_pixels_zeroes_region_and_clips():
    img = np.full((10, 12, 3), 255, dtype=np.uint8)
    out = mask_face_pixels(img, (-5, 2, 8, 50))  # clips to [0,w] and [0,h]
    # rows 2..10, cols 0..8 should be zero
    assert (out[2:10, 0:8] == 0).all()
    # outside region remains original
    assert (out[:2] == 255).all()
    assert (out[:, 8:] == 255).all()
    # original must be unchanged
    assert (img == 255).all()
 
 
class _FakeOnnxInput:
    def __init__(self, name: str):
        self.name = name
 
 
class _FakeSession:
    def __init__(self):
        self._inputs = [_FakeOnnxInput("input")]
        self.last_feed = None
 
    def get_inputs(self):
        return self._inputs
 
    def run(self, output_names, input_feed):
        self.last_feed = input_feed
        # deterministic, non-zero 512-dim vector
        v = np.arange(512, dtype=np.float32)[None, :]
        return [v]
 
 
def test_embedder_returns_512_dim_and_normalizes():
    sess = _FakeSession()
    emb = DocFrontBackgroundEmbedder(sess).embed_pil(
        Image.new("RGB", (64, 64), color=(10, 20, 30))
    )
    assert emb.shape == (512,)
    # normalized
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)
    # input tensor shape
    x = sess.last_feed["input"]
    assert x.shape == (1, 3, 224, 224)
    assert x.dtype == np.float32
 
 
def test_embedder_can_mask_face_bbox():
    sess = _FakeSession()
    img = Image.fromarray(np.full((100, 100, 3), 200, dtype=np.uint8), mode="RGB")
    _ = DocFrontBackgroundEmbedder(sess).embed_pil(
        img, face_bbox_xyxy=(10, 10, 90, 90)
    )
    x = sess.last_feed["input"]
    assert x.shape == (1, 3, 224, 224)
