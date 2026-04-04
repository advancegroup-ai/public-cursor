import numpy as np

from forgery_detection.embeddings.background_clip_onnx import (
    ClipBackgroundEmbedder,
    clip_image_to_input_tensor,
)


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(self) -> None:
        self.last_feed = None

    def get_inputs(self):
        return [_FakeInput("input")]

    def run(self, output_names, input_feed):
        self.last_feed = input_feed
        # Return a batch-1 embedding with a known norm.
        v = np.zeros((1, 512), dtype=np.float32)
        v[0, 0] = 3.0
        v[0, 1] = 4.0
        return [v]


def test_clip_preprocess_tensor_shape_and_dtype() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    t = clip_image_to_input_tensor(img)
    assert t.shape == (1, 3, 224, 224)
    assert t.dtype == np.float32


def test_embedder_normalizes() -> None:
    sess = _FakeSession()
    emb = ClipBackgroundEmbedder(sess, l2_normalize=True)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    v = emb.embed_bgr(img)
    assert v.shape == (512,)
    # vector was (3,4) => norm 5 => normalized
    assert np.isclose(float(v[0]), 0.6, atol=1e-6)
    assert np.isclose(float(v[1]), 0.8, atol=1e-6)

