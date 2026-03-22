from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class NpzVectorStore:
    ids: np.ndarray  # shape (n,), dtype=str or object
    vectors: np.ndarray  # shape (n, d), dtype=float32/float64

    @staticmethod
    def load(path: str | Path) -> NpzVectorStore:
        p = Path(path)
        with np.load(p, allow_pickle=False) as data:
            ids = data["ids"]
            vectors = data["vectors"]
        if ids.ndim != 1:
            raise ValueError(f"ids must be 1D, got shape {ids.shape}")
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"ids length {len(ids)} != vectors rows {vectors.shape[0]}")
        return NpzVectorStore(ids=ids, vectors=vectors)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, ids=self.ids, vectors=self.vectors)

    def subset(self, indices: Sequence[int] | np.ndarray) -> NpzVectorStore:
        idx = np.asarray(indices)
        return NpzVectorStore(ids=self.ids[idx], vectors=self.vectors[idx])

    def iter(self) -> Iterable[tuple[str, np.ndarray]]:
        for i, v in zip(self.ids.tolist(), self.vectors, strict=True):
            yield str(i), v
