from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .id_normalize import IdNormalization
from .npz_store import NpzVectorStore, _as_str_array


@dataclass(frozen=True)
class JoinReport:
    left_name: str
    right_name: str
    left_n: int
    right_n: int
    overlap_n: int
    overlap_ratio_left: float
    overlap_ratio_right: float
    left_only_n: int
    right_only_n: int
    examples_left_only: list[str]
    examples_right_only: list[str]


def join_on_normalized_ids(
    left: NpzVectorStore,
    right: NpzVectorStore,
    *,
    left_name: str = "left",
    right_name: str = "right",
    norm: IdNormalization | None = None,
    max_examples: int = 20,
) -> tuple[NpzVectorStore, NpzVectorStore, JoinReport]:
    """
    Inner-join two stores by normalized IDs.

    Returns (left_aligned, right_aligned, report) where both aligned stores share the same id order.
    """
    norm = norm or IdNormalization()

    l_ids_raw = left.ids.tolist()
    r_ids_raw = right.ids.tolist()
    l_norm = [norm.normalize(x) for x in l_ids_raw]
    r_norm = [norm.normalize(x) for x in r_ids_raw]

    l_map: dict[str, int] = {}
    for i, nid in enumerate(l_norm):
        if nid not in l_map:
            l_map[nid] = i
    r_map: dict[str, int] = {}
    for i, nid in enumerate(r_norm):
        if nid not in r_map:
            r_map[nid] = i

    overlap = sorted(set(l_map.keys()) & set(r_map.keys()))
    left_only = sorted(set(l_map.keys()) - set(r_map.keys()))
    right_only = sorted(set(r_map.keys()) - set(l_map.keys()))

    if overlap:
        l_idx = [l_map[nid] for nid in overlap]
        r_idx = [r_map[nid] for nid in overlap]
        # Use normalized ids in output to make downstream matching consistent
        out_ids = _as_str_array(overlap)
        left_aligned = NpzVectorStore(ids=out_ids, vectors=np.asarray(left.vectors)[l_idx])
        right_aligned = NpzVectorStore(ids=out_ids, vectors=np.asarray(right.vectors)[r_idx])
    else:
        out_ids = np.asarray([], dtype=str)
        left_aligned = NpzVectorStore(
            ids=out_ids,
            vectors=np.zeros((0, left.dim), dtype=left.vectors.dtype),
        )
        right_aligned = NpzVectorStore(
            ids=out_ids,
            vectors=np.zeros((0, right.dim), dtype=right.vectors.dtype),
        )

    report = JoinReport(
        left_name=left_name,
        right_name=right_name,
        left_n=left.size,
        right_n=right.size,
        overlap_n=int(len(overlap)),
        overlap_ratio_left=(len(overlap) / left.size) if left.size else 0.0,
        overlap_ratio_right=(len(overlap) / right.size) if right.size else 0.0,
        left_only_n=int(len(left_only)),
        right_only_n=int(len(right_only)),
        examples_left_only=left_only[:max_examples],
        examples_right_only=right_only[:max_examples],
    )
    return left_aligned, right_aligned, report

