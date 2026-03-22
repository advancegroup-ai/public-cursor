from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VectorStore:
    ids: list[str]
    vectors: np.ndarray  # (N, D) float32

    def __post_init__(self) -> None:
        if len(self.ids) != int(self.vectors.shape[0]):
            raise ValueError("ids length must match vectors rows")
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be 2D (N,D)")
        if self.vectors.dtype != np.float32:
            object.__setattr__(self, "vectors", self.vectors.astype(np.float32, copy=False))


def save_npz(path: str | Path, store: VectorStore) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, ids=np.asarray(store.ids, dtype=object), vectors=store.vectors)


def load_npz(path: str | Path) -> VectorStore:
    p = Path(path)
    with np.load(p, allow_pickle=True) as data:
        ids = [str(x) for x in data["ids"].tolist()]
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    return VectorStore(ids=ids, vectors=vectors)

