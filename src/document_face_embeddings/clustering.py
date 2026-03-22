from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .embedding import l2_normalize


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = l2_normalize(a)
    vb = l2_normalize(b)
    return float(np.dot(va, vb))


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D vectors matrix, got {x.shape}")
    xn = np.stack([l2_normalize(v) for v in x], axis=0)
    return xn @ xn.T


class UnionFind:
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
        elif self.rank[rb] < self.rank[ra]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray  # shape (N,), integer cluster ids [0..K-1]
    n_clusters: int


def threshold_clusters(vectors: np.ndarray, threshold: float) -> ClusterResult:
    """Connected-components clustering over cosine similarity >= threshold."""
    x = np.asarray(vectors, dtype=np.float32)
    n = x.shape[0]
    if n == 0:
        return ClusterResult(labels=np.zeros((0,), dtype=np.int32), n_clusters=0)
    sim = cosine_matrix(x)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                uf.union(i, j)
    root_to_label: dict[int, int] = {}
    labels = np.empty((n,), dtype=np.int32)
    next_label = 0
    for i in range(n):
        r = uf.find(i)
        if r not in root_to_label:
            root_to_label[r] = next_label
            next_label += 1
        labels[i] = root_to_label[r]
    return ClusterResult(labels=labels, n_clusters=next_label)


def cluster_sizes(labels: Iterable[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for l in labels:
        out[int(l)] = out.get(int(l), 0) + 1
    return out

