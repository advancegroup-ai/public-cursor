from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageRGB:
    """Simple image container for numpy RGB (H, W, 3), uint8."""

    array: np.ndarray

    @property
    def h(self) -> int:
        return int(self.array.shape[0])

    @property
    def w(self) -> int:
        return int(self.array.shape[1])


def load_rgb(path: str | Path) -> ImageRGB:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={arr.shape} for {path}")
    return ImageRGB(arr)


def save_rgb(path: str | Path, img: ImageRGB) -> None:
    Image.fromarray(img.array, mode="RGB").save(path)

