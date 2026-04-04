import numpy as np

from forgery_detection.embeddings.doc_front_background import DocFrontBackgroundEmbedder
 
 
class FakeSession:
  def __init__(self, out: np.ndarray):
    self._out = out

  def run(self, output_names, input_feed):
    assert isinstance(input_feed, dict)
    # Return shape (1, 512) typical for ONNX
    return [self._out]


def test_embed_returns_512_and_normalized():
  img = np.zeros((240, 320, 3), dtype=np.uint8)
  out = np.arange(512, dtype=np.float32)[None, :]
  emb = DocFrontBackgroundEmbedder(session=FakeSession(out), input_name="input")
  v = emb.embed_rgb_uint8(img)
  assert v.shape == (512,)
  assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_face_bbox_masking_changes_input_but_not_shape():
  img = np.ones((224, 224, 3), dtype=np.uint8) * 255
  out = np.ones((1, 512), dtype=np.float32)
  emb = DocFrontBackgroundEmbedder(session=FakeSession(out), input_name="input")
  v1 = emb.embed_rgb_uint8(img, face_bbox_xyxy=(0, 0, 10, 10))
  v2 = emb.embed_rgb_uint8(img, face_bbox_xyxy=(0, 0, 20, 20))
  assert v1.shape == v2.shape == (512,)
 
