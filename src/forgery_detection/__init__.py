"""Forgery detection utilities: embeddings, vector-store IO, joins, clustering."""

from .clustering import connected_components_cosine
from .id_normalization import IdNormalizer
from .join import JoinReport, join_on_normalized_ids
from .npz_store import NpzVectorStore

__all__ = [
    "IdNormalizer",
    "JoinReport",
    "NpzVectorStore",
    "connected_components_cosine",
    "join_on_normalized_ids",
]
