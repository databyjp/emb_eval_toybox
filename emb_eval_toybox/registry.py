"""Registry of available models and datasets."""

from typing import List, Dict, Tuple
import json
from pathlib import Path


def get_available_providers() -> List[Tuple[str, str]]:
    """Get list of available embedding providers.

    Returns:
        List of tuples (model_name, provider_type)
    """
    return [
        ("all-MiniLM-L6-v2", "sentence_transformers"),
        ("snowflake-arctic-embed:22m", "ollama"),
        ("snowflake-arctic-embed2", "ollama"),
        ("embed-english-v3.0", "cohere"),
    ]


def get_available_datasets() -> Dict[str, str]:
    """Get available evaluation datasets.

    Returns:
        Dictionary mapping dataset names to file paths
    """
    dataset_dir = Path("data")
    datasets = {}

    for file_path in dataset_dir.glob("synthetic_dataset_*.json"):
        with open(file_path) as f:
            data = json.load(f)
            name = data["metadata"]["name"]
            datasets[name] = str(file_path)

    return datasets
