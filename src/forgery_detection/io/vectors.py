from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class VectorTable:
    ids: list[str]
    vectors: np.ndarray  # (N,D) float32


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def load_vectors(path: str | Path) -> VectorTable:
    """
    Load vectors from:
      - .npy: raw (N,D) array; ids become 0..N-1
      - .npz: expects 'vectors' and optional 'ids'
      - .json: list[{"id": str, "vector": [...]}]

    Returns float32 vectors.
    """
    p = _as_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix == ".npy":
        vec = np.load(p)
        if vec.ndim != 2:
            raise ValueError(".npy must contain a 2D array")
        vec = vec.astype(np.float32, copy=False)
        ids = [str(i) for i in range(vec.shape[0])]
        return VectorTable(ids=ids, vectors=vec)

    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        if "vectors" not in data:
            raise ValueError(".npz must contain 'vectors'")
        vec = np.asarray(data["vectors"], dtype=np.float32)
        ids_arr = data["ids"] if "ids" in data else np.arange(vec.shape[0])
        ids = [str(x) for x in ids_arr.tolist()]
        return VectorTable(ids=ids, vectors=vec)

    if p.suffix == ".json":
        rows = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(".json must be a list of objects")
        ids: list[str] = []
        vecs: list[np.ndarray] = []
        for r in rows:
            if not isinstance(r, dict) or "id" not in r or "vector" not in r:
                raise ValueError("each row must have 'id' and 'vector'")
            ids.append(str(r["id"]))
            vecs.append(np.asarray(r["vector"], dtype=np.float32))
        if not vecs:
            raise ValueError("no vectors found")
        d = vecs[0].shape[0]
        if any(v.shape != (d,) for v in vecs):
            raise ValueError("all vectors must have the same dimension")
        return VectorTable(ids=ids, vectors=np.stack(vecs, axis=0))

    raise ValueError(f"unsupported vector file type: {p.suffix}")


def intersection_ids(a: Iterable[str], b: Iterable[str]) -> set[str]:
    return set(map(str, a)).intersection(set(map(str, b)))


def align_by_id(a: VectorTable, b: VectorTable) -> tuple[VectorTable, VectorTable]:
    """
    Returns aligned (a2, b2) containing only IDs present in both, in a's order.
    """
    b_index: dict[str, int] = {str(_id): i for i, _id in enumerate(b.ids)}
    keep_ids: list[str] = []
    a_rows: list[int] = []
    b_rows: list[int] = []
    for i, _id in enumerate(a.ids):
        k = str(_id)
        j = b_index.get(k)
        if j is None:
            continue
        keep_ids.append(k)
        a_rows.append(i)
        b_rows.append(j)
    if not keep_ids:
        return VectorTable(ids=[], vectors=np.zeros((0, a.vectors.shape[1]), dtype=np.float32)), VectorTable(
            ids=[], vectors=np.zeros((0, b.vectors.shape[1]), dtype=np.float32)
        )
    return (
        VectorTable(ids=keep_ids, vectors=np.asarray(a.vectors[a_rows], dtype=np.float32)),
        VectorTable(ids=keep_ids, vectors=np.asarray(b.vectors[b_rows], dtype=np.float32)),
    )


def to_json_rows(table: VectorTable) -> list[dict[str, Any]]:
    return [{"id": _id, "vector": table.vectors[i].tolist()} for i, _id in enumerate(table.ids)]

