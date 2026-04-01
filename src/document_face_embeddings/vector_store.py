from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class VectorStore:
    ids: np.ndarray  # dtype=str
    vectors: np.ndarray  # shape (N, D), float32

    def __post_init__(self) -> None:
        if self.ids.ndim != 1:
            raise ValueError("ids must be 1D")
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be 2D (N, D)")
        if self.ids.shape[0] != self.vectors.shape[0]:
            raise ValueError("ids and vectors must have same length")


def save_npz(path: str | Path, ids: Iterable[str], vectors: np.ndarray) -> None:
    p = Path(path)
    ids_arr = np.asarray(list(ids), dtype=str)
    vecs = np.asarray(vectors, dtype=np.float32)
    if vecs.ndim != 2:
        raise ValueError("vectors must be 2D (N, D)")
    if ids_arr.shape[0] != vecs.shape[0]:
        raise ValueError("ids and vectors must have same length")
    np.savez_compressed(p, ids=ids_arr, vectors=vecs)


def load_npz(path: str | Path) -> VectorStore:
    p = Path(path)
    data = np.load(p, allow_pickle=False)
    ids = np.asarray(data["ids"], dtype=str)
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    return VectorStore(ids=ids, vectors=vectors)

