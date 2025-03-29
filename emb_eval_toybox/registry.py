"""Registry of available models and datasets."""

from typing import List, Dict, Tuple

def get_available_providers() -> List[Tuple[str, str]]:
    """Get list of available embedding providers.

    Returns:
        List of tuples (model_name, provider_type)
    """
    return [
        ("all-MiniLM-L6-v2", "sentence_transformers"),
        ("snowflake-arctic-embed:22m", "ollama"),
        ("snowflake-arctic-embed2", "ollama"),
    ]

def get_available_datasets() -> Dict[str, str]:
    """Get available evaluation datasets.

    Returns:
        Dictionary mapping dataset names to file paths
    """
    return {
        "Trivia (Graded)": "data/synthetic_dataset_trivia_graded.json",
        "Trivia (Simple)": "data/synthetic_dataset_trivia.json",
        "Coffee": "data/synthetic_dataset_coffee.json",
    }
