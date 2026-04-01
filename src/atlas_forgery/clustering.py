from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .similarity import cosine_sim_matrix, l2_normalize


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray  # (N,) int64
    sizes: np.ndarray  # (K,) int64


def threshold_graph_clusters(vectors: np.ndarray, threshold: float) -> ClusterResult:
    """
    Connected-components clustering on a cosine-sim threshold graph.

    - vectors: (N, D)
    - threshold: edge exists if cos(i,j) >= threshold (i != j)
    """
    x = l2_normalize(np.asarray(vectors, dtype=np.float32))
    n = x.shape[0]
    if n == 0:
        return ClusterResult(labels=np.zeros((0,), dtype=np.int64), sizes=np.zeros((0,), dtype=np.int64))

    sim = cosine_sim_matrix(x, x)
    # Remove self loops
    np.fill_diagonal(sim, -1.0)
    adj = sim >= float(threshold)

    labels = -np.ones((n,), dtype=np.int64)
    cur = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        # BFS/stack
        stack = [i]
        labels[i] = cur
        while stack:
            u = stack.pop()
            neigh = np.where(adj[u])[0]
            for v in neigh.tolist():
                if labels[v] == -1:
                    labels[v] = cur
                    stack.append(v)
        cur += 1

    sizes = np.bincount(labels.astype(np.int64))
    return ClusterResult(labels=labels, sizes=sizes)

