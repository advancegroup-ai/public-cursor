from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NpzVectorStore:
    ids: list[str]
    vectors: np.ndarray  # shape: (n, d), dtype float32 preferred
 
    @staticmethod
    def load(path: str | Path) -> NpzVectorStore:
        data = np.load(str(path), allow_pickle=False)
        ids_arr = data["ids"]
        vecs = data["vectors"]
 
        if ids_arr.ndim != 1:
            raise ValueError(f"ids must be 1D, got shape={ids_arr.shape}")
        if vecs.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape={vecs.shape}")
        if vecs.shape[0] != ids_arr.shape[0]:
            raise ValueError(
                f"ids/vectors length mismatch: ids={ids_arr.shape[0]} vectors={vecs.shape[0]}"
            )
 
        ids = [str(x) for x in ids_arr.tolist()]
        vectors = vecs.astype(np.float32, copy=False)
        return NpzVectorStore(ids=ids, vectors=vectors)
 
    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Store ids as a unicode array to keep `allow_pickle=False` on load.
        ids = np.asarray(self.ids, dtype=str)
        np.savez_compressed(str(out), ids=ids, vectors=self.vectors)
 
    def subset(self, indices: Iterable[int]) -> NpzVectorStore:
        idx = np.asarray(list(indices), dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("indices must be 1D")
        ids = [self.ids[i] for i in idx.tolist()]
        vectors = self.vectors[idx]
        return NpzVectorStore(ids=ids, vectors=vectors)
