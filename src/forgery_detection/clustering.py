from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    n_clusters: int


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def cluster_by_cosine_threshold(vectors: np.ndarray, threshold: float) -> ClusterResult:
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got {vectors.shape}")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [-1, 1]")

    n = int(vectors.shape[0])
    if n == 0:
        return ClusterResult(labels=np.empty((0,), dtype=np.int64), n_clusters=0)

    unit = _l2_normalize(vectors.astype(np.float32, copy=False))
    sim = unit @ unit.T

    dsu = _DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i, j]) >= threshold:
                dsu.union(i, j)

    roots = np.array([dsu.find(i) for i in range(n)], dtype=np.int64)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    return ClusterResult(labels=labels.astype(np.int64), n_clusters=int(unique_roots.shape[0]))
