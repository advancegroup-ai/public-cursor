from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray  # shape (N,), int
    clusters: List[List[int]]  # list of index lists


def cluster_by_cosine_threshold(vectors: np.ndarray, threshold: float) -> ClusterResult:
    """
    Connected-components clustering in a graph where edge(i,j)=1 iff cosine(v_i,v_j) >= threshold.

    Assumes vectors are L2-normalized; if not, cosine is computed with explicit normalization.
    """
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected (N,D) vectors, got shape {x.shape}")
    n = x.shape[0]
    if n == 0:
        return ClusterResult(labels=np.zeros((0,), dtype=np.int64), clusters=[])

    # Normalize defensively
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x = x / norms

    sim = x @ x.T  # (N,N)
    adj = sim >= float(threshold)
    np.fill_diagonal(adj, True)

    labels = -np.ones((n,), dtype=np.int64)
    clusters: List[List[int]] = []

    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        # BFS/DFS
        stack = [i]
        labels[i] = cid
        members = []
        while stack:
            u = stack.pop()
            members.append(u)
            nbrs = np.where(adj[u])[0]
            for v in nbrs:
                if labels[v] == -1:
                    labels[v] = cid
                    stack.append(int(v))
        clusters.append(sorted(members))
        cid += 1

    return ClusterResult(labels=labels, clusters=clusters)

