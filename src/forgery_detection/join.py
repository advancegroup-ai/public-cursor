from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .id_normalization import IdNormalizer
from .npz_store import NpzVectorStore


@dataclass(frozen=True, slots=True)
class JoinReport:
    left_total: int
    right_total: int
    left_unique_norm: int
    right_unique_norm: int
    overlap_norm: int
    overlap_left_frac: float
    overlap_right_frac: float


def _norm_map(ids: Iterable[str], normalizer: IdNormalizer) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, raw in enumerate(ids):
        k = normalizer.normalize(str(raw))
        # keep first occurrence to ensure deterministic join
        out.setdefault(k, idx)
    return out


def join_on_normalized_ids(
    left: NpzVectorStore,
    right: NpzVectorStore,
    *,
    normalizer: IdNormalizer | None = None,
) -> tuple[NpzVectorStore, NpzVectorStore, JoinReport]:
    """
    Inner-join two vector stores by normalized IDs, returning aligned stores.

    - Normalization is applied to stringified IDs.
    - If multiple raw IDs normalize to the same key, only the *first* is kept.
    """
    norm = normalizer or IdNormalizer()
    lmap = _norm_map(left.ids.tolist(), norm)
    rmap = _norm_map(right.ids.tolist(), norm)

    lkeys = set(lmap.keys())
    rkeys = set(rmap.keys())
    overlap = sorted(lkeys & rkeys)

    li = np.array([lmap[k] for k in overlap], dtype=np.int64)
    ri = np.array([rmap[k] for k in overlap], dtype=np.int64)

    left_joined = left.subset(li)
    right_joined = right.subset(ri)

    report = JoinReport(
        left_total=len(left.ids),
        right_total=len(right.ids),
        left_unique_norm=len(lkeys),
        right_unique_norm=len(rkeys),
        overlap_norm=len(overlap),
        overlap_left_frac=(len(overlap) / max(1, len(lkeys))),
        overlap_right_frac=(len(overlap) / max(1, len(rkeys))),
    )
    return left_joined, right_joined, report
