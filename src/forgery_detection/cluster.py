from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
 
from forgery_detection.math_utils import l2_normalize_rows
 
 
@dataclass(frozen=True)
class Cluster:
    cluster_id: int
    member_indices: list[int]
 
 
def cluster_by_cosine_threshold(
    ids: list[str],
    vectors: np.ndarray,
    *,
    threshold: float,
) -> list[Cluster]:
    if threshold < -1.0 or threshold > 1.0:
        raise ValueError("threshold must be in [-1, 1]")
    if len(ids) != vectors.shape[0]:
        raise ValueError("ids/vectors length mismatch")
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    n = vectors.shape[0]
    if n == 0:
        return []
 
    v = l2_normalize_rows(vectors.astype(np.float32, copy=False))
    sim = v @ v.T  # (n,n)
    adj = sim >= float(threshold)
    np.fill_diagonal(adj, True)
 
    visited = np.zeros(n, dtype=bool)
    clusters: list[Cluster] = []
    cid = 0
 
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        members: list[int] = []
        while stack:
            i = stack.pop()
            members.append(i)
            neigh = np.flatnonzero(adj[i] & ~visited)
            if neigh.size:
                visited[neigh] = True
                stack.extend(neigh.tolist())
        clusters.append(Cluster(cluster_id=cid, member_indices=sorted(members)))
        cid += 1
 
    return clusters
