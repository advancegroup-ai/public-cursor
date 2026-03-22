from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class NamedVectors:
    ids: List[str]
    vectors: np.ndarray  # shape (N, D)

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {k: self.vectors[i] for i, k in enumerate(self.ids)}


def load_vectors(path: str, *, id_col: str = "id", vector_col: str = "vector") -> NamedVectors:
    """
    Load vectors from:
      - .npz: expects arrays "ids" and "vectors"
      - .npy: expects array of shape (N,D) and uses index as id
      - .json: list of {id:..., vector:[...]} or dict {id:[...]}
      - .csv: columns (id_col, vector_col) where vector_col is json list
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".npz":
        data = np.load(p, allow_pickle=True)
        ids = [str(x) for x in data["ids"].tolist()]
        vecs = np.asarray(data["vectors"], dtype=np.float32)
        return NamedVectors(ids=ids, vectors=vecs)
    if suf == ".npy":
        vecs = np.asarray(np.load(p, allow_pickle=True), dtype=np.float32)
        if vecs.ndim != 2:
            raise ValueError(f".npy must be 2D, got shape {vecs.shape}")
        ids = [str(i) for i in range(vecs.shape[0])]
        return NamedVectors(ids=ids, vectors=vecs)
    if suf == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            ids = [str(k) for k in obj.keys()]
            vecs = np.stack([np.asarray(obj[k], dtype=np.float32) for k in obj.keys()], axis=0)
            return NamedVectors(ids=ids, vectors=vecs)
        if isinstance(obj, list):
            ids = [str(it[id_col]) for it in obj]
            vecs = np.stack([np.asarray(it[vector_col], dtype=np.float32) for it in obj], axis=0)
            return NamedVectors(ids=ids, vectors=vecs)
        raise ValueError("Unsupported JSON format for vectors")
    if suf == ".csv":
        ids: List[str] = []
        vecs: List[np.ndarray] = []
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.append(str(row[id_col]))
                vecs.append(np.asarray(json.loads(row[vector_col]), dtype=np.float32))
        return NamedVectors(ids=ids, vectors=np.stack(vecs, axis=0))
    raise ValueError(f"Unsupported vector file type: {suf}")


def intersect_by_id(a: NamedVectors, b: NamedVectors) -> Tuple[NamedVectors, NamedVectors]:
    a_map = {k: i for i, k in enumerate(a.ids)}
    b_map = {k: i for i, k in enumerate(b.ids)}
    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    a_idx = [a_map[k] for k in common]
    b_idx = [b_map[k] for k in common]
    return (
        NamedVectors(ids=common, vectors=a.vectors[a_idx]),
        NamedVectors(ids=common, vectors=b.vectors[b_idx]),
    )

