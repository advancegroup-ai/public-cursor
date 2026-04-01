from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class VectorStore:
    ids: np.ndarray  # shape (N,), dtype=str
    vectors: np.ndarray  # shape (N, D), float32

    def __post_init__(self) -> None:
        if self.ids.ndim != 1:
            raise ValueError("ids must be 1D")
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        if self.vectors.shape[0] != self.ids.shape[0]:
            raise ValueError("ids and vectors must have same length")


def save_npz(path: str | Path, ids: Sequence[str], vectors: np.ndarray) -> None:
    path = Path(path)
    ids_arr = np.asarray(list(ids), dtype=str)
    vecs = np.asarray(vectors, dtype=np.float32)
    np.savez_compressed(path, ids=ids_arr, vectors=vecs)


def load_npz(path: str | Path) -> VectorStore:
    data = np.load(Path(path), allow_pickle=False)
    ids = data["ids"].astype(str)
    vectors = data["vectors"].astype(np.float32)
    return VectorStore(ids=ids, vectors=vectors)


def intersect_ids(a: VectorStore, b: VectorStore) -> tuple[VectorStore, VectorStore]:
    a_map = {str(i): idx for idx, i in enumerate(a.ids.tolist())}
    b_map = {str(i): idx for idx, i in enumerate(b.ids.tolist())}
    common: list[str] = sorted(set(a_map).intersection(b_map))
    a_idx = np.asarray([a_map[i] for i in common], dtype=int)
    b_idx = np.asarray([b_map[i] for i in common], dtype=int)
    return (
        VectorStore(ids=np.asarray(common, dtype=str), vectors=a.vectors[a_idx]),
        VectorStore(ids=np.asarray(common, dtype=str), vectors=b.vectors[b_idx]),
    )

