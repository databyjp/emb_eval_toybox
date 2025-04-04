import fasttext
import numpy as np
from typing import List, Union
from huggingface_hub import hf_hub_download
from .base import EmbeddingProvider


class FastTextProvider(EmbeddingProvider):
    """Provider for FastText embeddings."""

    def __init__(
        self,
        model_name: str = "facebook/fasttext-en-vectors",
    ):
        """Initialize the provider with a specific model.

        Args:
            model_name: Name of the FastText model to use (Hugging Face model ID)
        """
        self._model_name = model_name
        # Download and load the model
        model_path = hf_hub_download(repo_id=model_name, filename="model.bin")
        self._model = fasttext.load_model(model_path)
        # Get embedding dimension by encoding a test string
        self._dimension = len(self._model.get_word_vector("test"))

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using FastText.

        Args:
            texts: Single text or list of texts to encode

        Returns:
            numpy.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Split text into words and get average of word vectors
            words = text.split()
            if not words:
                # If empty text, use zero vector
                embeddings.append(np.zeros(self._dimension))
                continue

            word_vectors = [self._model.get_word_vector(word) for word in words]
            avg_vector = np.mean(word_vectors, axis=0)
            embeddings.append(avg_vector)

        return np.array(embeddings)

    @property
    def name(self) -> str:
        return f"fasttext-{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension
