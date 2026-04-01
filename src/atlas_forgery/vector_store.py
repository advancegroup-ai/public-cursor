from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VectorStore:
    ids: np.ndarray
    vectors: np.ndarray

    def validate(self) -> None:
        if self.ids.ndim != 1:
            raise ValueError("ids must be a 1D array")
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if len(self.ids) != len(self.vectors):
            raise ValueError("ids and vectors must have the same first dimension")


def save_npz(path: str | Path, ids: np.ndarray, vectors: np.ndarray) -> None:
    store = VectorStore(ids=ids, vectors=vectors)
    store.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, ids=ids, vectors=vectors)


def load_npz(path: str | Path) -> VectorStore:
    data = np.load(Path(path), allow_pickle=False)
    store = VectorStore(ids=data["ids"], vectors=data["vectors"])
    store.validate()
    return store
