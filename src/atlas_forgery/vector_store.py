from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorStore:
    ids: np.ndarray  # dtype=str/object
    vectors: np.ndarray  # float32, shape (N, D)
    meta: dict[str, Any] | None = None


def save_npz(path: str | Path, ids: list[str] | np.ndarray, vectors: np.ndarray, **meta: Any) -> None:
    p = Path(path)
    ids_arr = np.asarray(ids)
    vec = np.asarray(vectors, dtype=np.float32)
    if vec.ndim != 2:
        raise ValueError(f"vectors must be 2D, got {vec.shape}")
    np.savez_compressed(p, ids=ids_arr, vectors=vec, meta=np.asarray([meta], dtype=object))


def load_npz(path: str | Path) -> VectorStore:
    p = Path(path)
    with np.load(p, allow_pickle=True) as data:
        ids = data["ids"]
        vectors = data["vectors"].astype(np.float32, copy=False)
        meta_arr = data["meta"] if "meta" in data else None
        meta = None
        if meta_arr is not None and len(meta_arr) > 0:
            meta = dict(meta_arr[0])
    return VectorStore(ids=ids, vectors=vectors, meta=meta)

