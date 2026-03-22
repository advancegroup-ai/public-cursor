from __future__ import annotations

import numpy as np


def _cosine_similarity_matrix(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    v = vectors / norms
    return v @ v.T


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def cluster_by_cosine_threshold(vectors: np.ndarray, threshold: float) -> list[int]:
    """
    Single-linkage clustering by cosine similarity threshold.

    Two points are connected if cosine_sim >= threshold; clusters are connected components.
    Returns list of integer labels aligned with input order.
    """
    vecs = np.asarray(vectors, dtype=np.float32)
    n = vecs.shape[0]
    if n == 0:
        return []
    if vecs.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {vecs.shape}")
    if not (-1.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between -1 and 1")

    sim = _cosine_similarity_matrix(vecs)
    uf = _UnionFind(n)
    for i in range(n):
        row = sim[i]
        for j in range(i + 1, n):
            if row[j] >= threshold:
                uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    root_to_label: dict[int, int] = {}
    labels: list[int] = []
    next_label = 0
    for r in roots:
        if r not in root_to_label:
            root_to_label[r] = next_label
            next_label += 1
        labels.append(root_to_label[r])
    return labels

