from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / max(n, eps)
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array; got shape={x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return a @ b.T


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray  # shape: (N,), int
    clusters: list[list[int]]  # indices for each cluster label


def threshold_graph_clusters(vectors: np.ndarray, threshold: float) -> ClusterResult:
    """
    Build an undirected graph where edges connect pairs with cosine >= threshold,
    then return connected components as clusters.
    """
    x = l2_normalize(vectors)
    n = int(x.shape[0])
    sims = x @ x.T
    adj = sims >= float(threshold)
    np.fill_diagonal(adj, True)

    labels = -np.ones(n, dtype=np.int32)
    clusters: list[list[int]] = []

    current = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        # BFS/DFS over boolean adjacency matrix
        stack = [i]
        labels[i] = current
        members: list[int] = []
        while stack:
            u = stack.pop()
            members.append(u)
            nbrs = np.flatnonzero(adj[u])
            for v in nbrs.tolist():
                if labels[v] == -1:
                    labels[v] = current
                    stack.append(v)
        clusters.append(sorted(members))
        current += 1

    return ClusterResult(labels=labels, clusters=clusters)

