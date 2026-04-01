from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x = x / norms
    return x @ x.T


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray  # shape (N,), int
    n_clusters: int


def threshold_graph_clusters(vectors: np.ndarray, threshold: float) -> ClusterResult:
    """
    Cluster by connected components in a cosine-similarity graph (>= threshold).
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    n = vectors.shape[0]
    if n == 0:
        return ClusterResult(labels=np.zeros((0,), dtype=np.int32), n_clusters=0)

    sim = cosine_similarity_matrix(vectors)
    adj = sim >= float(threshold)
    np.fill_diagonal(adj, True)

    labels = -np.ones((n,), dtype=np.int32)
    cur = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cur
        while stack:
            u = stack.pop()
            nbrs = np.flatnonzero(adj[u])
            for v in nbrs:
                if labels[v] == -1:
                    labels[v] = cur
                    stack.append(int(v))
        cur += 1
    return ClusterResult(labels=labels, n_clusters=int(cur))

