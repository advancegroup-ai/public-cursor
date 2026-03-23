from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


def id_normalize(x: str) -> str:
    """
    Normalize IDs for joining across sources.

    - Strips whitespace
    - Lowercases
    - Keeps alnum only
    """
    if x is None:
        return ""
    x = str(x).strip().lower()
    return "".join(ch for ch in x if ch.isalnum())


@dataclass(frozen=True)
class JoinReport:
    left_count: int
    right_count: int
    left_unique: int
    right_unique: int
    intersection: int
    left_only: int
    right_only: int


def join_on_normalized_ids(left_ids: Sequence[str], right_ids: Sequence[str]) -> JoinReport:
    left = [id_normalize(x) for x in left_ids]
    right = [id_normalize(x) for x in right_ids]
    left_set = set(x for x in left if x)
    right_set = set(x for x in right if x)
    inter = left_set & right_set
    return JoinReport(
        left_count=len(left_ids),
        right_count=len(right_ids),
        left_unique=len(left_set),
        right_unique=len(right_set),
        intersection=len(inter),
        left_only=len(left_set - right_set),
        right_only=len(right_set - left_set),
    )


def mask_bbox_zero_rgb(img: np.ndarray, bbox_xyxy: Sequence[int | float]) -> np.ndarray:
    """
    Zero out pixels in bbox region (xyxy) for an RGB/BGR image array.

    Returns a copy; does not mutate input.
    """
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 image, got shape={img.shape}")

    x0, y0, x1, y1 = bbox_xyxy
    h, w = img.shape[:2]
    x0 = int(max(0, min(w, round(x0))))
    x1 = int(max(0, min(w, round(x1))))
    y0 = int(max(0, min(h, round(y0))))
    y1 = int(max(0, min(h, round(y1))))
    if x1 <= x0 or y1 <= y0:
        return img.copy()

    out = img.copy()
    out[y0:y1, x0:x1, :3] = 0
    return out


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """
    Cosine similarity matrix.

    - a: (N,D)
    - b: (M,D) or None -> b=a
    Returns: (N,M)
    """
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("a must be 2D")
    if b is None:
        b = a
    b = np.asarray(b, dtype=np.float32)
    if b.ndim != 2:
        raise ValueError("b must be 2D")
    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have same feature dimension")
    a_n = _l2_normalize_rows(a)
    b_n = _l2_normalize_rows(b)
    return a_n @ b_n.T


def cluster_by_cosine_threshold(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """
    Simple connected-components clustering where an edge exists if cosine_sim >= threshold.

    Intended for analysis (not optimized for very large N).
    """
    x = np.asarray(vectors, dtype=np.float32)
    n = x.shape[0]
    if n == 0:
        return []
    sims = cosine_sim_matrix(x)
    # Include self edges, then BFS components.
    visited = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        q = [i]
        visited[i] = True
        comp: list[int] = []
        while q:
            u = q.pop()
            comp.append(u)
            neigh = np.where(sims[u] >= threshold)[0]
            for v in neigh.tolist():
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        clusters.append(sorted(comp))
    clusters.sort(key=len, reverse=True)
    return clusters


def batched(iterable: Iterable[int], batch_size: int) -> Iterable[list[int]]:
    batch: list[int] = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
