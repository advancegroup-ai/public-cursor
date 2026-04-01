from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .id_normalization import IdNormalizer


@dataclass(frozen=True)
class JoinReport:
    left_count: int
    right_count: int
    left_unique_norm: int
    right_unique_norm: int
    overlap_unique_norm: int
    overlap_rows: int

    def to_dict(self) -> dict[str, int]:
        return {
            "left_count": self.left_count,
            "right_count": self.right_count,
            "left_unique_norm": self.left_unique_norm,
            "right_unique_norm": self.right_unique_norm,
            "overlap_unique_norm": self.overlap_unique_norm,
            "overlap_rows": self.overlap_rows,
        }


@dataclass(frozen=True)
class NpzVectorStore:
    ids: np.ndarray  # (N,) dtype=object|str
    vectors: np.ndarray  # (N, D) float32/float64

    @property
    def dim(self) -> int:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape={self.vectors.shape}")
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    @staticmethod
    def load(path: str | Path) -> NpzVectorStore:
        p = Path(path)
        data = np.load(p, allow_pickle=True)
        if "ids" not in data or "vectors" not in data:
            raise ValueError(f"NPZ missing required arrays: ids,vectors at {p}")
        ids = data["ids"]
        vectors = data["vectors"]
        if ids.ndim != 1:
            raise ValueError(f"ids must be 1D, got shape={ids.shape}")
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape={vectors.shape}")
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(
                f"ids and vectors length mismatch: {ids.shape[0]} vs {vectors.shape[0]}"
            )
        return NpzVectorStore(ids=ids, vectors=vectors)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, ids=self.ids, vectors=self.vectors)

    def subset(self, indices: Sequence[int]) -> NpzVectorStore:
        idx = np.asarray(indices, dtype=np.int64)
        return NpzVectorStore(ids=self.ids[idx], vectors=self.vectors[idx])


def _build_first_index(norm_ids: Iterable[str]) -> dict[str, int]:
    first: dict[str, int] = {}
    for i, nid in enumerate(norm_ids):
        if nid not in first:
            first[nid] = i
    return first


def join_on_normalized_ids(
    left: NpzVectorStore,
    right: NpzVectorStore,
    *,
    normalizer: IdNormalizer | None = None,
) -> tuple[NpzVectorStore, NpzVectorStore, JoinReport]:
    """
    Align two stores on normalized ids.

    - Uses "first occurrence wins" strategy per normalized id on each side.
    - Returns aligned (left,right) stores with the SAME length and SAME normalized-id ordering.
    """
    normalizer = normalizer or IdNormalizer()

    left_ids = [str(x) for x in left.ids.tolist()]
    right_ids = [str(x) for x in right.ids.tolist()]
    left_norm = normalizer.normalize_many(left_ids)
    right_norm = normalizer.normalize_many(right_ids)

    left_first = _build_first_index(left_norm)
    right_first = _build_first_index(right_norm)

    overlap_norm = sorted(set(left_first).intersection(right_first))
    left_idx = [left_first[n] for n in overlap_norm]
    right_idx = [right_first[n] for n in overlap_norm]

    left_joined = left.subset(left_idx)
    right_joined = right.subset(right_idx)

    report = JoinReport(
        left_count=len(left),
        right_count=len(right),
        left_unique_norm=len(set(left_norm)),
        right_unique_norm=len(set(right_norm)),
        overlap_unique_norm=len(overlap_norm),
        overlap_rows=len(overlap_norm),
    )
    return left_joined, right_joined, report

