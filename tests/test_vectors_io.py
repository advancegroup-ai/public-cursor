import json
from pathlib import Path

import numpy as np

from forgery_detection.io.vectors import align_by_id, load_vectors, VectorTable


def test_load_vectors_npy(tmp_path: Path):
    p = tmp_path / "x.npy"
    x = np.random.randn(5, 7).astype(np.float32)
    np.save(p, x)
    vt = load_vectors(p)
    assert vt.vectors.shape == (5, 7)
    assert vt.ids[0] == "0"


def test_load_vectors_json_and_align(tmp_path: Path):
    p = tmp_path / "x.json"
    rows = [{"id": "a", "vector": [1.0, 0.0]}, {"id": "b", "vector": [0.0, 1.0]}]
    p.write_text(json.dumps(rows), encoding="utf-8")
    a = load_vectors(p)
    b = VectorTable(ids=["b", "c"], vectors=np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    a2, b2 = align_by_id(a, b)
    assert a2.ids == ["b"]
    assert b2.ids == ["b"]
    assert a2.vectors.shape == (1, 2)

