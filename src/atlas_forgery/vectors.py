from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorStore:
    ids: np.ndarray  # shape (N,), dtype=object or str
    vectors: np.ndarray  # shape (N, D), float32
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.ids.ndim != 1:
            raise ValueError(f"ids must be 1D; got {self.ids.shape}")
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D; got {self.vectors.shape}")
        if self.vectors.shape[0] != self.ids.shape[0]:
            raise ValueError("ids and vectors must have same first dimension")


def save_npz(path: str | Path, store: VectorStore) -> None:
    p = Path(path)
    payload: dict[str, Any] = {
        "ids": store.ids.astype(object),
        "vectors": store.vectors.astype(np.float32),
    }
    if store.meta is not None:
        payload["meta_json"] = np.array([json.dumps(store.meta)], dtype=object)
    np.savez_compressed(p, **payload)


def load_npz(path: str | Path) -> VectorStore:
    p = Path(path)
    with np.load(p, allow_pickle=True) as z:
        ids = z["ids"]
        vectors = z["vectors"].astype(np.float32)
        meta = None
        if "meta_json" in z:
            meta = json.loads(str(z["meta_json"][0]))
        return VectorStore(ids=ids, vectors=vectors, meta=meta)

