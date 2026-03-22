from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_similarity_matrix(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("vectors must be 2D (N, D)")
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    x = x / norms
    return x @ x.T


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray  # shape (N,), int64, -1 means unassigned (shouldn't happen here)
    n_clusters: int


def threshold_connected_components(sim: np.ndarray, threshold: float) -> ClusteringResult:
    """Cluster by graph connectivity where edge exists if sim[i,j] >= threshold.

    This is a simple, deterministic baseline for template-reuse style linking.
    """
    s = np.asarray(sim, dtype=np.float32)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("sim must be square (N, N)")
    n = s.shape[0]
    adj = s >= float(threshold)
    np.fill_diagonal(adj, True)

    labels = np.full((n,), -1, dtype=np.int64)
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        # BFS/DFS
        stack = [i]
        labels[i] = cluster_id
        while stack:
            u = stack.pop()
            neighbors = np.flatnonzero(adj[u])
            for v in neighbors:
                if labels[v] == -1:
                    labels[v] = cluster_id
                    stack.append(int(v))
        cluster_id += 1

    return ClusteringResult(labels=labels, n_clusters=int(cluster_id))


def cluster_size_stats(labels: np.ndarray) -> dict[str, int]:
    lab = np.asarray(labels, dtype=np.int64)
    if lab.ndim != 1:
        raise ValueError("labels must be 1D")
    uniq, counts = np.unique(lab, return_counts=True)
    sizes = counts.tolist()
    return {
        "n_clusters": int(uniq.shape[0]),
        "n_items": int(lab.shape[0]),
        "max_cluster_size": int(max(sizes) if sizes else 0),
        "min_cluster_size": int(min(sizes) if sizes else 0),
    }

