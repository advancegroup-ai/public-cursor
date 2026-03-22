from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class EmbeddingTable:
    ids: list[str]
    vectors: np.ndarray  # shape: (n, d)
    meta: Mapping[str, Any]


def load_embedding_npz(path: str | Path) -> EmbeddingTable:
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    # Accept a few common conventions
    ids_key = "ids" if "ids" in data else ("id" if "id" in data else None)
    vec_key = (
        "vectors"
        if "vectors" in data
        else ("embeddings" if "embeddings" in data else ("X" if "X" in data else None))
    )
    if ids_key is None or vec_key is None:
        raise ValueError(
            f"{p}: expected keys like ids+vectors (got {sorted(list(data.keys()))})"
        )
    ids = [str(x) for x in data[ids_key].tolist()]
    vectors = np.asarray(data[vec_key])
    if vectors.ndim != 2:
        raise ValueError(f"{p}: vectors must be 2D, got shape={vectors.shape}")
    if len(ids) != vectors.shape[0]:
        raise ValueError(
            f"{p}: ids length {len(ids)} != vectors rows {vectors.shape[0]}"
        )
    meta = {k: data[k].tolist() for k in data.files if k not in {ids_key, vec_key}}
    return EmbeddingTable(ids=ids, vectors=vectors, meta=meta)

