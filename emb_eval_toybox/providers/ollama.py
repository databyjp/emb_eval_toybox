import requests
from typing import List, Union
import numpy as np
from .base import EmbeddingProvider


class OllamaProvider(EmbeddingProvider):
    """Provider for Ollama embeddings."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize the provider with a specific model.

        Args:
            model_name: Name of the Ollama model to use
            base_url: URL of the Ollama instance
        """
        self._model_name = model_name
        self._base_url = base_url
        # Get embedding dimension by encoding a dummy text
        self._dimension = self.encode("test").shape[0]

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Ollama.

        Args:
            texts: Single text or list of texts to encode

        Returns:
            numpy.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model_name, "prompt": text},
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])

        return np.array(embeddings)

    @property
    def name(self) -> str:
        return f"ollama-{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension
