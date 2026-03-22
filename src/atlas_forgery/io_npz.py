from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .id_normalization import normalize_signature_id


@dataclass(frozen=True)
class EmbeddingTable:
    ids: list[str]
    vectors: np.ndarray  # shape: (N, D), float32/float64

    def normalized_id_map(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i, raw in enumerate(self.ids):
            out[normalize_signature_id(raw).normalized] = i
        return out


def _coerce_ids(arr: Any) -> list[str]:
    if arr is None:
        return []
    if isinstance(arr, (list, tuple)):
        return [str(x) for x in arr]
    a = np.asarray(arr)
    if a.ndim == 0:
        return [str(a.item())]
    return [str(x) for x in a.tolist()]


def load_embeddings_npz(path: str | Path) -> dict[str, EmbeddingTable]:
    """
    Load embedding artifacts from .npz.

    Supported patterns:
    - {name}_ids + {name}_vectors
    - ids + vectors (single table, returned under key "default")
    """
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    keys = set(data.files)

    tables: dict[str, EmbeddingTable] = {}

    if "ids" in keys and "vectors" in keys:
        ids = _coerce_ids(data["ids"])
        vecs = np.asarray(data["vectors"])
        tables["default"] = EmbeddingTable(ids=ids, vectors=vecs)
        return tables

    # Scan for pairs.
    for k in list(keys):
        if not k.endswith("_ids"):
            continue
        prefix = k[: -len("_ids")]
        vec_key = f"{prefix}_vectors"
        if vec_key not in keys:
            continue
        ids = _coerce_ids(data[k])
        vecs = np.asarray(data[vec_key])
        tables[prefix] = EmbeddingTable(ids=ids, vectors=vecs)

    if not tables:
        raise ValueError(f"No embedding tables found in {p} (keys={sorted(keys)})")
    return tables

