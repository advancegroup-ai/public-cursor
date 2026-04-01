from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def _as_str_array(ids: Iterable[object]) -> np.ndarray:
    arr = np.asarray(list(ids))
    # Force string dtype for stable serialization
    return arr.astype(str)


@dataclass(frozen=True)
class NpzVectorStore:
    ids: np.ndarray  # shape (N,), dtype=str
    vectors: np.ndarray  # shape (N, D), dtype=float32/float64

    def __post_init__(self) -> None:
        if self.ids.ndim != 1:
            raise ValueError(f"ids must be 1D, got shape={self.ids.shape}")
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape={self.vectors.shape}")
        if self.ids.shape[0] != self.vectors.shape[0]:
            raise ValueError("ids and vectors must have same length")

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    def id_set(self) -> set[str]:
        return set(map(str, self.ids.tolist()))

    def reindex_by_id(self) -> dict[str, int]:
        idx: dict[str, int] = {}
        for i, sid in enumerate(self.ids.tolist()):
            s = str(sid)
            # Keep first occurrence deterministically
            if s not in idx:
                idx[s] = i
        return idx

    def select(self, ids: Iterable[str]) -> NpzVectorStore:
        idx_map = self.reindex_by_id()
        sel_ids: list[str] = []
        sel_vecs: list[np.ndarray] = []
        for sid in ids:
            i = idx_map.get(str(sid))
            if i is None:
                continue
            sel_ids.append(str(self.ids[i]))
            sel_vecs.append(self.vectors[i])
        if not sel_ids:
            return NpzVectorStore(
                ids=np.asarray([], dtype=str),
                vectors=np.zeros((0, self.dim), dtype=self.vectors.dtype),
            )
        return NpzVectorStore(ids=_as_str_array(sel_ids), vectors=np.stack(sel_vecs, axis=0))

    def save(self, path: str) -> None:
        np.savez_compressed(path, ids=_as_str_array(self.ids), vectors=self.vectors)

    @staticmethod
    def load(path: str) -> NpzVectorStore:
        data = np.load(path, allow_pickle=False)
        if "ids" not in data or "vectors" not in data:
            raise ValueError("NPZ must contain 'ids' and 'vectors'")
        ids = np.asarray(data["ids"]).astype(str)
        vectors = np.asarray(data["vectors"])
        return NpzVectorStore(ids=ids, vectors=vectors)
