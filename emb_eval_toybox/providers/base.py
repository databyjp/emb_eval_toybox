from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for the given texts.

        Args:
            texts: Single text or list of texts to encode

        Returns:
            numpy.ndarray: Array of embeddings
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the embedding provider."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embeddings."""
        pass
