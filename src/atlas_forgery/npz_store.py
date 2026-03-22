from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NpzVectorStore:
    ids: list[str]
    vectors: np.ndarray  # [N, D]

    def __post_init__(self) -> None:
        v = np.asarray(self.vectors)
        if v.ndim != 2:
            raise ValueError(f"vectors must be 2D [N,D], got shape={v.shape}")
        if len(self.ids) != v.shape[0]:
            raise ValueError(f"ids length {len(self.ids)} != vectors rows {v.shape[0]}")

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.array(self.ids, dtype=object), vectors=self.vectors)

    @staticmethod
    def load(path: str | Path) -> NpzVectorStore:
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        ids = [str(x) for x in data["ids"].tolist()]
        vectors = np.asarray(data["vectors"])
        return NpzVectorStore(ids=ids, vectors=vectors)

    def to_dict(self) -> dict[str, np.ndarray]:
        return {sid: self.vectors[i] for i, sid in enumerate(self.ids)}

    def subset(self, ids: list[str]) -> NpzVectorStore:
        index = {sid: i for i, sid in enumerate(self.ids)}
        keep_idx = [index[sid] for sid in ids if sid in index]
        keep_ids = [self.ids[i] for i in keep_idx]
        keep_vec = self.vectors[keep_idx]
        return NpzVectorStore(ids=keep_ids, vectors=keep_vec)
