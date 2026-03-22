from __future__ import annotations

from typing import Protocol

import numpy as np


class FaceEmbedder(Protocol):
    """Embed a single aligned face crop into a 512-d float vector."""

    def embed_aligned_rgb(self, face_rgb_u8_112: np.ndarray) -> np.ndarray:
        """
        Args:
            face_rgb_u8_112: uint8 array shaped (112,112,3) in RGB.
        Returns:
            1D float32 embedding vector, typically 512-d. Caller may L2-normalize.
        """

