from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray  # shape (N,), int64
    n_clusters: int


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def cluster_by_cosine_threshold(vectors: np.ndarray, threshold: float) -> ClusteringResult:
    """
    Connected-components clustering over a cosine-similarity threshold graph.

    - vectors: (N, D)
    - threshold: connect i<->j if cos(v_i, v_j) >= threshold
    """
    if threshold < -1.0 or threshold > 1.0:
        raise ValueError("threshold must be within [-1, 1]")

    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim != 2 or v.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array")

    v = _l2_normalize(v)
    n = v.shape[0]

    # Similarity matrix (N,N). For typical batch sizes in analysis this is OK.
    sim = v @ v.T
    adj = sim >= float(threshold)
    np.fill_diagonal(adj, True)

    labels = -np.ones((n,), dtype=np.int64)
    cur = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cur
        while stack:
            u = stack.pop()
            nbrs = np.flatnonzero(adj[u])
            for w in nbrs:
                if labels[w] == -1:
                    labels[w] = cur
                    stack.append(int(w))
        cur += 1

    return ClusteringResult(labels=labels, n_clusters=int(cur))

