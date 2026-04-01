from __future__ import annotations
 
from dataclasses import dataclass
from typing import Protocol
 
import numpy as np
 
 
class Embedder(Protocol):
    dim: int
 
    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Returns a 1D float32 vector of length `dim`.
        """
 
 
@dataclass(frozen=True)
class DeterministicStubEmbedder:
    """
    Deterministic baseline embedder for wiring/tests (NOT a model).
    """
 
    dim: int = 512
 
    def embed(self, image: np.ndarray) -> np.ndarray:
        x = np.asarray(image)
        h = int(np.abs(x).sum()) % (2**32)
        rng = np.random.default_rng(h)
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        return v
