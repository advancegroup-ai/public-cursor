from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorStore:
    ids: list[str]
    vectors: np.ndarray  # shape: (N, D)
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D (N,D); got shape={self.vectors.shape}")
        if len(self.ids) != int(self.vectors.shape[0]):
            raise ValueError(
                f"ids length must match vectors N; ids={len(self.ids)} N={self.vectors.shape[0]}"
            )


def save_npz(path: str | Path, store: VectorStore) -> None:
    path = Path(path)
    meta_json = json.dumps(store.meta or {}, ensure_ascii=False)
    np.savez_compressed(
        path,
        ids=np.array(store.ids, dtype=object),
        vectors=store.vectors.astype(np.float32, copy=False),
        meta=np.array([meta_json], dtype=object),
    )


def load_npz(path: str | Path) -> VectorStore:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    ids = [str(x) for x in data["ids"].tolist()]
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    meta_raw = data.get("meta", None)
    meta: dict[str, Any] | None
    if meta_raw is None:
        meta = None
    else:
        meta_json = meta_raw.tolist()[0]
        meta = json.loads(meta_json) if meta_json else {}
    return VectorStore(ids=ids, vectors=vectors, meta=meta)

