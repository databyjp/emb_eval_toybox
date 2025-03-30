"""
Embedding providers for different models and services.
"""

from typing import Dict, Type
from .base import EmbeddingProvider
from .sentence_transformers import SentenceTransformersProvider
from .ollama import OllamaProvider
from .cohere import CohereProvider

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformersProvider",
    "OllamaProvider",
    "CohereProvider",
    "create_provider",
]

# Registry of available provider types
PROVIDER_REGISTRY: Dict[str, Type[EmbeddingProvider]] = {
    "sentence_transformers": SentenceTransformersProvider,
    "ollama": OllamaProvider,
    "cohere": CohereProvider,
}

def create_provider(provider_type: str, provider_name: str) -> EmbeddingProvider:
    """Create an embedding provider instance.

    Args:
        provider_type: Type of provider ("sentence_transformers", "ollama", or "cohere")
        provider_name: Name of the model to use

    Returns:
        An instance of the appropriate EmbeddingProvider

    Raises:
        ValueError: If the provider type is unknown
    """
    if provider_type not in PROVIDER_REGISTRY:
        available_types = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown provider type: {provider_type}. Available types: {available_types}"
        )

    provider_class = PROVIDER_REGISTRY[provider_type]
    return provider_class(provider_name)
