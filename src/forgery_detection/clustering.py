from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ClusterResult:
    labels: np.ndarray  # shape (n,), int64
    n_clusters: int


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def connected_components_cosine(vectors: np.ndarray, *, threshold: float) -> ClusterResult:
    """
    Simple, deterministic clustering via connected-components over cosine similarity.

    Complexity is O(n^2) for simplicity; this is intended for small/medium batches.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got {vectors.ndim}D")
    n = vectors.shape[0]
    if n == 0:
        return ClusterResult(labels=np.zeros((0,), dtype=np.int64), n_clusters=0)

    x = _l2_normalize(vectors.astype(np.float32, copy=False))
    sim = x @ x.T
    adj = sim >= float(threshold)
    np.fill_diagonal(adj, True)

    labels = -np.ones((n,), dtype=np.int64)
    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            j = stack.pop()
            nbrs = np.flatnonzero(adj[j] & (labels == -1))
            if nbrs.size:
                labels[nbrs] = cid
                stack.extend(nbrs.tolist())
        cid += 1

    return ClusterResult(labels=labels, n_clusters=cid)
