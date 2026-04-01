from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .npz_store import NpzVectorStore


def _normalize_id(x: str) -> str:
    # Conservative normalization: trim and lowercase; keep internal characters.
    return x.strip().lower()


@dataclass(frozen=True)
class JoinReport:
    left_total: int
    right_total: int
    left_unique_norm: int
    right_unique_norm: int
    intersection_unique_norm: int
    left_only_unique_norm: int
    right_only_unique_norm: int
    left_rows_joined: int
    right_rows_joined: int


def join_on_normalized_ids(
    left: NpzVectorStore, right: NpzVectorStore
) -> tuple[NpzVectorStore, NpzVectorStore, JoinReport]:
    left_norm = np.array([_normalize_id(x) for x in left.ids.tolist()], dtype="U")
    right_norm = np.array([_normalize_id(x) for x in right.ids.tolist()], dtype="U")

    left_unique = set(left_norm.tolist())
    right_unique = set(right_norm.tolist())
    inter = left_unique & right_unique

    # Map normalized id -> first index (stable)
    left_first: dict[str, int] = {}
    for i, nid in enumerate(left_norm.tolist()):
        left_first.setdefault(nid, i)
    right_first: dict[str, int] = {}
    for i, nid in enumerate(right_norm.tolist()):
        right_first.setdefault(nid, i)

    inter_sorted = sorted(inter)
    left_idx = np.array([left_first[nid] for nid in inter_sorted], dtype=np.int64)
    right_idx = np.array([right_first[nid] for nid in inter_sorted], dtype=np.int64)

    left_joined = NpzVectorStore(ids=left.ids[left_idx], vectors=left.vectors[left_idx])
    right_joined = NpzVectorStore(ids=right.ids[right_idx], vectors=right.vectors[right_idx])

    report = JoinReport(
        left_total=len(left),
        right_total=len(right),
        left_unique_norm=len(left_unique),
        right_unique_norm=len(right_unique),
        intersection_unique_norm=len(inter),
        left_only_unique_norm=len(left_unique - right_unique),
        right_only_unique_norm=len(right_unique - left_unique),
        left_rows_joined=int(left_idx.shape[0]),
        right_rows_joined=int(right_idx.shape[0]),
    )
    return left_joined, right_joined, report
