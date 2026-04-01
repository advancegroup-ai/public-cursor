from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    # assume already l2-normalized; still be robust
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.clip(norms, 1e-12, None)
    return v @ v.T


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray  # shape (N,), int
    n_clusters: int


def threshold_connected_components(sim: np.ndarray, threshold: float) -> ClusteringResult:
    """Graph CCs where edge exists if sim[i,j] >= threshold (i!=j)."""
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("sim must be square")
    n = sim.shape[0]
    visited = np.zeros((n,), dtype=bool)
    labels = np.full((n,), -1, dtype=int)
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        labels[i] = cluster_id
        while stack:
            u = stack.pop()
            neigh = np.where(sim[u] >= threshold)[0]
            for v in neigh.tolist():
                if v == u:
                    continue
                if not visited[v]:
                    visited[v] = True
                    labels[v] = cluster_id
                    stack.append(v)
        cluster_id += 1
    return ClusteringResult(labels=labels, n_clusters=cluster_id)


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    labels = np.asarray(labels, dtype=int)
    out: dict[int, int] = {}
    for lab in labels.tolist():
        out[lab] = out.get(lab, 0) + 1
    return out

