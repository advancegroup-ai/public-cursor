from __future__ import annotations

import numpy as np


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return a @ b.T

