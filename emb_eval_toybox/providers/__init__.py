"""
Embedding providers for different models and services.
"""

from .base import EmbeddingProvider
from .sentence_transformers import SentenceTransformersProvider

__all__ = ["EmbeddingProvider", "SentenceTransformersProvider"]
