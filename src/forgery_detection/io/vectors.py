from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class VectorSet:
  ids: List[str]
  vectors: np.ndarray  # (N, D)


def load_vectors(path: str) -> VectorSet:
  p = Path(path)
  suffix = p.suffix.lower()

  if suffix == ".npy":
    arr = np.load(p)
    if arr.ndim != 2:
      raise ValueError("expected npy to contain a 2D array (N, D)")
    ids = [str(i) for i in range(arr.shape[0])]
    return VectorSet(ids=ids, vectors=np.asarray(arr, dtype=np.float32))

  if suffix == ".npz":
    data = np.load(p)
    if "vectors" not in data:
      raise ValueError("npz must contain 'vectors' array")
    vecs = np.asarray(data["vectors"], dtype=np.float32)
    ids = data["ids"].tolist() if "ids" in data else [str(i) for i in range(vecs.shape[0])]
    ids = [str(x) for x in ids]
    return VectorSet(ids=ids, vectors=vecs)

  if suffix == ".json":
    obj = json.loads(p.read_text())
    if isinstance(obj, dict) and "vectors" in obj:
      vecs = np.asarray(obj["vectors"], dtype=np.float32)
      ids = [str(x) for x in obj.get("ids", list(range(vecs.shape[0])))]
      return VectorSet(ids=ids, vectors=vecs)
    raise ValueError("json must be an object with keys: vectors[, ids]")

  if suffix == ".csv":
    with p.open(newline="") as f:
      reader = csv.DictReader(f)
      if reader.fieldnames is None:
        raise ValueError("empty csv")
      if "id" not in reader.fieldnames:
        raise ValueError("csv must have an 'id' column")
      vec_cols = [c for c in reader.fieldnames if c != "id"]
      if not vec_cols:
        raise ValueError("csv must have vector columns besides 'id'")
      ids: List[str] = []
      vecs: List[List[float]] = []
      for row in reader:
        ids.append(str(row["id"]))
        vecs.append([float(row[c]) for c in vec_cols])
    arr = np.asarray(vecs, dtype=np.float32)
    return VectorSet(ids=ids, vectors=arr)

  raise ValueError(f"unsupported vector file type: {suffix}")
 
