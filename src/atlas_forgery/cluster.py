from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _cosine_sim_matrix(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"Expected 2D array [N,D], got shape={v.shape}")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.maximum(norms, eps)
    return v @ v.T


def threshold_graph_clusters(
    vectors: np.ndarray,
    threshold: float,
    *,
    labels: Iterable[str] | None = None,
) -> list[list[int]]:
    """
    Build an undirected graph with edges where cosine_sim >= threshold,
    then return connected components as lists of indices.
    """
    sims = _cosine_sim_matrix(vectors)
    n = sims.shape[0]
    adj = sims >= float(threshold)
    np.fill_diagonal(adj, True)

    seen = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []

    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            neigh = np.where(adj[u])[0]
            for v in neigh:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        clusters.append(sorted(comp))

    # stable ordering: largest clusters first, then lexicographic indices
    clusters.sort(key=lambda c: (-len(c), c))
    return clusters
