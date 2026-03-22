from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairMetrics:
    total_pairs: int
    linked_pairs: int
    linkage_rate: float


def pairwise_linkage_metrics(
    doc_vectors: np.ndarray,
    face_vectors: np.ndarray,
    doc_sim_threshold: float = 0.92,
    face_sim_upper: float = 0.45,
) -> PairMetrics:
    """
    Flag pairs where doc backgrounds are similar but faces are dissimilar.
    """
    d = np.asarray(doc_vectors, dtype=np.float32)
    f = np.asarray(face_vectors, dtype=np.float32)
    if d.shape[0] != f.shape[0]:
        raise ValueError("doc and face vectors length mismatch")
    n = d.shape[0]
    if n < 2:
        return PairMetrics(total_pairs=0, linked_pairs=0, linkage_rate=0.0)

    d = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-12, None)
    f = f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-12, None)

    doc_sim = d @ d.T
    face_sim = f @ f.T

    linked = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if doc_sim[i, j] >= doc_sim_threshold and face_sim[i, j] <= face_sim_upper:
                linked += 1
    rate = linked / total if total else 0.0
    return PairMetrics(total_pairs=total, linked_pairs=linked, linkage_rate=rate)

