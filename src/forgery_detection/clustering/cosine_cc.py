from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray  # int64 [N], -1 not used (all assigned)
    n_clusters: int


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def cluster_by_cosine_threshold(vectors: np.ndarray, threshold: float) -> ClusterResult:
    """
    Connected-components clustering where an edge exists if cosine(u, v) >= threshold.

    Notes:
    - This is O(N^2) and meant for small-medium N (e.g., case batches).
    - Input vectors are L2-normalized internally.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be within [0, 1]")
    x = np.asarray(vectors)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D vectors, got shape={x.shape}")
    n = x.shape[0]
    if n == 0:
        return ClusterResult(labels=np.zeros((0,), dtype=np.int64), n_clusters=0)
    x = _l2_normalize_rows(x)

    sim = x @ x.T
    adj = sim >= float(threshold)
    np.fill_diagonal(adj, True)

    labels = np.full((n,), -1, dtype=np.int64)
    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            u = stack.pop()
            neigh = np.flatnonzero(adj[u])
            for v in neigh:
                if labels[v] == -1:
                    labels[v] = cid
                    stack.append(int(v))
        cid += 1
    return ClusterResult(labels=labels, n_clusters=cid)

