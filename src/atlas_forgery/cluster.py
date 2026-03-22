from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray  # (N,) int
    clusters: list[list[int]]  # indices per component


def _cosine_sim_matrix(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.divide(v, n, out=np.zeros_like(v), where=n != 0)
    return v @ v.T


def threshold_graph_clusters(vectors: np.ndarray, threshold: float) -> ClusterResult:
    """
    Cluster by building an undirected graph with edges where cosine >= threshold,
    then return connected components.
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    n = vectors.shape[0]
    if n == 0:
        return ClusterResult(labels=np.zeros((0,), dtype=int), clusters=[])

    sim = _cosine_sim_matrix(vectors)
    adj = sim >= float(threshold)
    np.fill_diagonal(adj, True)

    labels = -np.ones((n,), dtype=int)
    clusters: list[list[int]] = []
    cur = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        # BFS/DFS on adjacency matrix
        stack = [i]
        labels[i] = cur
        comp: list[int] = []
        while stack:
            j = stack.pop()
            comp.append(j)
            neigh = np.where(adj[j])[0]
            for k in neigh:
                if labels[k] == -1:
                    labels[k] = cur
                    stack.append(k)
        clusters.append(sorted(comp))
        cur += 1

    return ClusterResult(labels=labels, clusters=clusters)

