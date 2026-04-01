from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NpzVectorStore:
    ids: np.ndarray  # shape (N,), dtype str (unicode)
    vectors: np.ndarray  # shape (N, D), float32/float64

    @property
    def dim(self) -> int:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {self.vectors.shape}")
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    @staticmethod
    def load(path: str | Path) -> NpzVectorStore:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            ids = data["ids"]
            vectors = data["vectors"]

        if ids.ndim != 1:
            raise ValueError(f"ids must be 1D, got shape {ids.shape}")
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(f"vectors rows {vectors.shape[0]} must match ids {ids.shape[0]}")

        # Enforce unicode ids so downstream joins don’t trip over bytes/object dtypes.
        if ids.dtype.kind != "U":
            ids = ids.astype("U")
        return NpzVectorStore(ids=ids, vectors=vectors)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = self.ids
        if ids.dtype.kind != "U":
            ids = ids.astype("U")
        np.savez_compressed(path, ids=ids, vectors=self.vectors)
