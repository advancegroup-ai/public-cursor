from __future__ import annotations
 
import numpy as np
 
 
def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)
