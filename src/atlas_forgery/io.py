from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorStore:
    ids: np.ndarray  # shape (N,), dtype=str
    vectors: np.ndarray  # shape (N, D), float32
    meta: dict[str, Any] | None = None


def save_npz(
    path: str | Path, ids: np.ndarray, vectors: np.ndarray, meta: dict | None = None
) -> None:
    path = Path(path)
    if ids.ndim != 1:
        raise ValueError("ids must be 1D")
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    if len(ids) != len(vectors):
        raise ValueError("ids and vectors must have same length")

    meta_json = None if meta is None else json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(
        path,
        ids=ids.astype(str),
        vectors=vectors.astype(np.float32),
        meta=meta_json,
    )


def load_npz(path: str | Path) -> VectorStore:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        ids = z["ids"].astype(str)
        vectors = z["vectors"].astype(np.float32)
        meta_json = z["meta"].tolist() if "meta" in z.files else None
        meta = None if meta_json in (None, "", "None") else json.loads(meta_json)
    return VectorStore(ids=ids, vectors=vectors, meta=meta)

