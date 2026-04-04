"""Document embeddings utilities (with optional face exclusion)."""

from .embedding import ImageEmbedder, MaskedEmbedder
from .io import load_image_rgb

__all__ = ["ImageEmbedder", "MaskedEmbedder", "load_image_rgb"]

