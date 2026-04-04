from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
  denom = np.linalg.norm(x, axis=1, keepdims=True)
  return x / np.maximum(denom, eps)


def cluster_by_cosine_threshold(vectors: np.ndarray, threshold: float) -> List[List[int]]:
  """
  Cluster vectors using connected-components on a cosine similarity graph.

  - vectors: (N, D)
  - threshold: edge exists if cosine_sim >= threshold

  Returns clusters as lists of indices.
  """
  vecs = np.asarray(vectors, dtype=np.float32)
  if vecs.ndim != 2:
    raise ValueError("expected 2D array (N, D)")
  n = vecs.shape[0]
  if n == 0:
    return []

  vecs = _l2_normalize_rows(vecs)
  sims = vecs @ vecs.T
  adj = sims >= float(threshold)
  np.fill_diagonal(adj, True)

  seen = np.zeros(n, dtype=bool)
  clusters: List[List[int]] = []

  for i in range(n):
    if seen[i]:
      continue
    stack = [i]
    seen[i] = True
    comp = []
    while stack:
      u = stack.pop()
      comp.append(u)
      nbrs = np.flatnonzero(adj[u] & (~seen))
      if nbrs.size:
        seen[nbrs] = True
        stack.extend(nbrs.tolist())
    clusters.append(sorted(comp))

  clusters.sort(key=len, reverse=True)
  return clusters


def top_pairs_by_cosine(vectors: np.ndarray, k: int = 20) -> List[Tuple[int, int, float]]:
  """Utility for reporting most similar pairs (excluding diagonal)."""
  vecs = _l2_normalize_rows(np.asarray(vectors, dtype=np.float32))
  sims = vecs @ vecs.T
  np.fill_diagonal(sims, -np.inf)
  n = sims.shape[0]
  flat_idx = np.argpartition(sims.ravel(), -min(k, n * n - n))[-min(k, n * n - n) :]
  pairs = []
  for idx in flat_idx:
    i = int(idx // n)
    j = int(idx % n)
    if i < j:
      pairs.append((i, j, float(sims[i, j])))
  pairs.sort(key=lambda t: t[2], reverse=True)
  return pairs[:k]
