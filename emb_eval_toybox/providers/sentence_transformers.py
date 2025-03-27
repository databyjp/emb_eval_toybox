from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import EmbeddingProvider


class SentenceTransformersProvider(EmbeddingProvider):
    """Provider for sentence-transformers models."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        trust_remote_code: bool = False,
    ):
        """Initialize the provider with a specific model.

        Args:
            model_name: Name of the sentence-transformers model to use
            trust_remote_code: Whether to trust remote code (needed for some models like Arctic)
        """
        self._model = SentenceTransformer(
            model_name, trust_remote_code=trust_remote_code
        )
        self._model_name = model_name
        # Get embedding dimension by encoding a dummy text
        self._dimension = self._model.encode("test").shape[0]

    def encode(
        self, texts: Union[str, List[str]], prompt_name: Optional[str] = None
    ) -> np.ndarray:
        """Generate embeddings using sentence-transformers.

        Args:
            texts: Single text or list of texts to encode
            prompt_name: Optional prompt name for models that support it (e.g., "query" for Arctic)
        """
        return self._model.encode(texts, prompt_name=prompt_name)

    @property
    def name(self) -> str:
        return f"sentence-transformers-{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension
