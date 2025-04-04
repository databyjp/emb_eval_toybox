"""Registry of available models and datasets."""

from typing import List, Dict, Tuple
import json
from pathlib import Path


def get_available_providers() -> List[Tuple[str, str]]:
    """Get list of available embedding providers.

    Returns:
        List of tuples (model_name, provider_type)
    """
    # Default models if no provider-specific configurations exist
    providers = [
        ("all-MiniLM-L6-v2", "sentence_transformers"),
        ("all-minilm:22m", "ollama"),
        ("snowflake-arctic-embed:22m", "ollama"),
        ("snowflake-arctic-embed2", "ollama"),
        ("embed-english-v3.0", "cohere"),
        ("facebook/fasttext-en-vectors", "fasttext"),
    ]

    return providers


def get_available_datasets() -> Dict[str, str]:
    """Get available evaluation datasets.

    Returns:
        Dictionary mapping dataset names to file paths
    """
    dataset_dir = Path("data")
    datasets = {}

    if dataset_dir.exists():
        for file_path in dataset_dir.glob("synthetic_dataset_*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    name = data["metadata"]["name"]
                    datasets[name] = str(file_path)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load dataset from {file_path}: {e}")

    return datasets
