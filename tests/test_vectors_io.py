import json
from pathlib import Path

import numpy as np

from forgery_detection.io.vectors import NamedVectors, intersect_by_id, load_vectors


def test_intersect_by_id_orders_common_ids():
    a = NamedVectors(ids=["b", "a"], vectors=np.array([[2.0], [1.0]], dtype=np.float32))
    b = NamedVectors(ids=["a", "c", "b"], vectors=np.array([[10.0], [30.0], [20.0]], dtype=np.float32))
    ai, bi = intersect_by_id(a, b)
    assert ai.ids == ["a", "b"]
    assert bi.ids == ["a", "b"]
    assert ai.vectors.tolist() == [[1.0], [2.0]]
    assert bi.vectors.tolist() == [[10.0], [20.0]]


def test_load_vectors_json_dict(tmp_path: Path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"x": [1, 2], "y": [3, 4]}), encoding="utf-8")
    named = load_vectors(str(p))
    assert named.ids == ["x", "y"]
    assert named.vectors.shape == (2, 2)

