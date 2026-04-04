from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    # assumes vectors are already L2-normalized (or close enough)
    return vectors @ vectors.T


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray  # shape (N,), ints 0..K-1
    num_clusters: int

    def clusters(self) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in range(self.num_clusters)]
        for i, lab in enumerate(self.labels.tolist()):
            out[lab].append(i)
        return out


def cluster_by_cosine_threshold(vectors: Sequence[np.ndarray], threshold: float) -> ClusterResult:
    """
    Build a graph with edges where cosine_sim >= threshold; return connected components.
    """
    vecs = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors], axis=0)
    if vecs.ndim != 2:
        raise ValueError("vectors must be a sequence of 1D arrays")
    if vecs.shape[0] == 0:
        return ClusterResult(labels=np.zeros((0,), dtype=np.int32), num_clusters=0)

    sims = _cosine_similarity_matrix(vecs)
    n = sims.shape[0]
    visited = np.zeros((n,), dtype=bool)
    labels = np.full((n,), -1, dtype=np.int32)
    cid = 0

    for i in range(n):
        if visited[i]:
            continue
        # BFS/DFS over thresholded adjacency
        stack = [i]
        visited[i] = True
        labels[i] = cid
        while stack:
            u = stack.pop()
            neigh = np.where((sims[u] >= threshold) & (~visited))[0]
            for v in neigh.tolist():
                visited[v] = True
                labels[v] = cid
                stack.append(v)
        cid += 1

    return ClusterResult(labels=labels, num_clusters=cid)

