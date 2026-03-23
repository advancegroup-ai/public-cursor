from __future__ import annotations
 
from dataclasses import dataclass
 
from forgery_detection.id_normalization import IdNormalizer
from forgery_detection.vector_store import NpzVectorStore
 
 
@dataclass(frozen=True)
class JoinReport:
    left_count: int
    right_count: int
    overlap_count: int
    overlap_rate_left: float
    overlap_rate_right: float
 
 
@dataclass(frozen=True)
class JoinedStores:
    left: NpzVectorStore
    right: NpzVectorStore
    report: JoinReport
 
 
def join_on_normalized_ids(
    left: NpzVectorStore,
    right: NpzVectorStore,
    *,
    normalizer: IdNormalizer | None = None,
) -> JoinedStores:
    norm = normalizer or IdNormalizer()
 
    left_norm_to_idx: dict[str, int] = {}
    for i, rid in enumerate(left.ids):
        k = norm.normalize(rid)
        if k not in left_norm_to_idx:
            left_norm_to_idx[k] = i
 
    right_norm_to_idx: dict[str, int] = {}
    for i, rid in enumerate(right.ids):
        k = norm.normalize(rid)
        if k not in right_norm_to_idx:
            right_norm_to_idx[k] = i
 
    overlap_keys = sorted(set(left_norm_to_idx).intersection(right_norm_to_idx))
    left_idx = [left_norm_to_idx[k] for k in overlap_keys]
    right_idx = [right_norm_to_idx[k] for k in overlap_keys]
 
    joined_left = left.subset(left_idx)
    joined_right = right.subset(right_idx)
 
    overlap = len(overlap_keys)
    rep = JoinReport(
        left_count=len(left.ids),
        right_count=len(right.ids),
        overlap_count=overlap,
        overlap_rate_left=(overlap / len(left.ids)) if left.ids else 0.0,
        overlap_rate_right=(overlap / len(right.ids)) if right.ids else 0.0,
    )
    return JoinedStores(left=joined_left, right=joined_right, report=rep)
