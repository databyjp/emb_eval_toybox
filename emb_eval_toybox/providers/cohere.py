import os
import cohere
import numpy as np
from typing import List, Union, Optional
from .base import EmbeddingProvider


class CohereProvider(EmbeddingProvider):
    """Provider for Cohere embeddings."""

    def __init__(
        self,
        model_name: str = "embed-english-v3.0",
        input_type: str = "search_query",
        api_key: Optional[str] = None,
    ):
        """Initialize the provider with a specific model.

        Args:
            model_name: Name of the Cohere model to use
            input_type: Type of input ("search_query" or "search_document")
            api_key: Cohere API key. If None, reads from COHERE_API_KEY environment variable
        """
        self._model_name = model_name
        self._input_type = input_type
        api_key = api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key not found. Set COHERE_API_KEY environment variable."
            )

        self._client = cohere.Client(api_key=api_key)
        # Get embedding dimension by encoding a test string
        test_embedding = self._client.embed(
            texts=["test"],
            model=model_name,
            input_type=input_type,
            embedding_types=["float"],
        ).embeddings.float[0]
        self._dimension = len(test_embedding)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Cohere.

        Args:
            texts: Single text or list of texts to encode

        Returns:
            numpy.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        response = self._client.embed(
            texts=texts,
            model=self._model_name,
            input_type=self._input_type,
            embedding_types=["float"],
        )
        return np.array(response.embeddings.float)

    @property
    def name(self) -> str:
        return f"cohere-{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension
