from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorFile:
    ids: list[str]
    vectors: np.ndarray  # float32 [N,D]


def _ensure_2d_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    return x.astype(np.float32, copy=False)


def load_vectors(path: str | Path) -> VectorFile:
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
        arr = _ensure_2d_f32(arr)
        ids = [str(i) for i in range(arr.shape[0])]
        return VectorFile(ids=ids, vectors=arr)

    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        if "vectors" not in data:
            raise ValueError("NPZ must contain 'vectors'")
        vecs = _ensure_2d_f32(data["vectors"])
        if "ids" in data:
            ids_arr = data["ids"]
            ids = [str(x) for x in ids_arr.tolist()]
        else:
            ids = [str(i) for i in range(vecs.shape[0])]
        if len(ids) != vecs.shape[0]:
            raise ValueError("ids length does not match vectors rows")
        return VectorFile(ids=ids, vectors=vecs)

    if p.suffix == ".json":
        obj: Any = json.loads(p.read_text(encoding="utf-8"))
        ids = [str(x) for x in obj["ids"]]
        vecs = _ensure_2d_f32(np.asarray(obj["vectors"], dtype=np.float32))
        if len(ids) != vecs.shape[0]:
            raise ValueError("ids length does not match vectors rows")
        return VectorFile(ids=ids, vectors=vecs)

    raise ValueError(f"Unsupported vector file type: {p.suffix}")


def save_vectors(path: str | Path, ids: list[str], vectors: np.ndarray) -> None:
    p = Path(path)
    vectors = _ensure_2d_f32(vectors)
    if len(ids) != vectors.shape[0]:
        raise ValueError("ids length does not match vectors rows")

    if p.suffix == ".npz":
        np.savez_compressed(p, ids=np.array(ids, dtype=object), vectors=vectors)
        return
    if p.suffix == ".json":
        payload = {"ids": ids, "vectors": vectors.tolist()}
        p.write_text(json.dumps(payload), encoding="utf-8")
        return
    if p.suffix == ".npy":
        np.save(p, vectors)
        return
    raise ValueError(f"Unsupported output type: {p.suffix}")


def align_by_id(left: VectorFile, right: VectorFile) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Align two vector sets by id intersection.
    Returns (ids, left_vectors, right_vectors) with same row order.
    """
    left_idx = {k: i for i, k in enumerate(left.ids)}
    right_idx = {k: i for i, k in enumerate(right.ids)}
    common = sorted(set(left_idx).intersection(right_idx))
    lv = left.vectors[[left_idx[k] for k in common]]
    rv = right.vectors[[right_idx[k] for k in common]]
    return common, lv, rv

