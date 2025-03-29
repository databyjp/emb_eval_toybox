"""Core evaluation functionality for embedding models."""

from pathlib import Path
import numpy as np
from typing import List, Dict, Any
from .data.dataset import SearchDataset
from .providers import SentenceTransformersProvider, OllamaProvider, CohereProvider


def calculate_dcg(relevance_scores: list[int], k: int = None) -> float:
    """Calculate Discounted Cumulative Gain.

    Args:
        relevance_scores: List of relevance scores (0, 1, or 2)
        k: Number of results to consider. If None, uses all results.

    Returns:
        DCG score
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]

    gains = [2**score - 1 for score in relevance_scores]
    dcg = gains[0]
    for i, gain in enumerate(gains[1:], 1):
        dcg += gain / np.log2(i + 1)

    return dcg


def calculate_ndcg(
    actual_scores: list[int], ideal_scores: list[int], k: int = None
) -> float:
    """Calculate Normalized Discounted Cumulative Gain.

    Args:
        actual_scores: List of relevance scores in predicted order
        ideal_scores: List of relevance scores in ideal order
        k: Number of results to consider

    Returns:
        NDCG score
    """
    dcg = calculate_dcg(actual_scores, k)
    idcg = calculate_dcg(ideal_scores, k)
    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_recall(
    predicted_indices: List[int], true_relevant: List[int], k: int = None
) -> tuple[float, float]:
    """Calculate precision and recall at k.

    Args:
        predicted_indices: List of predicted document indices
        true_relevant: List of indices of truly relevant documents
        k: Number of results to consider. If None, uses all results.

    Returns:
        Tuple of (precision@k, recall@k)
    """
    if k is not None:
        predicted_indices = predicted_indices[:k]

    # Convert to sets for intersection
    true_relevant_set = set(true_relevant)
    predicted_set = set(predicted_indices)

    # Calculate true positives (correctly predicted relevant documents)
    true_positives = len(true_relevant_set.intersection(predicted_set))

    # Precision = true positives / total predicted (at k)
    precision = true_positives / len(predicted_indices) if predicted_indices else 0.0

    # Recall = true positives / total relevant
    recall = true_positives / len(true_relevant_set) if true_relevant_set else 1.0

    return precision, recall


def evaluate_basic(
    dataset_path: str,
    provider_name: str = "all-MiniLM-L6-v2",
    provider_type: str = "sentence_transformers",
    k_values: List[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate an embedding model using basic metrics (precision/recall).

    Args:
        dataset_path: Path to the dataset JSON file
        provider_name: Name of the model to use
        provider_type: Type of provider ("sentence_transformers" or "ollama")
        k_values: List of k values for computing metrics. If None, uses [3, 5, 10]

    Returns:
        List of dictionaries containing evaluation results for each query
    """
    dataset = SearchDataset(dataset_path)
    if dataset.evaluation_type != "basic":
        raise ValueError(f"Dataset {dataset_path} is not suitable for basic evaluation")

    # Number of results to retrieve is the maximum k value
    num_results = max(k_values) if k_values else 10

    # Initialize the embedding provider
    if provider_type == "sentence_transformers":
        provider = SentenceTransformersProvider(provider_name)
    elif provider_type == "ollama":
        provider = OllamaProvider(provider_name)
    elif provider_type == "cohere":
        provider = CohereProvider(provider_name)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

    # Generate embeddings
    query_embeddings = provider.encode(dataset.queries)
    doc_embeddings = provider.encode(dataset.documents)
    similarities = np.dot(query_embeddings, doc_embeddings.T)

    results = []
    for query_idx, (query, relevant_docs) in enumerate(dataset):
        top_k_indices = np.argsort(similarities[query_idx])[-num_results:][::-1]

        # Calculate precision/recall metrics
        precision_recall_scores = {}
        true_relevant_indices = [
            i for i, score in enumerate(dataset.relevance[query_idx])
            if score > 0
        ]

        for k in k_values:
            precision, recall = calculate_precision_recall(
                top_k_indices.tolist(), true_relevant_indices, k
            )
            precision_recall_scores[k] = {"precision": precision, "recall": recall}

        # Get relevant documents with scores (same as original)
        all_docs_with_scores = [
            (doc, score)
            for doc, score in zip(dataset.documents, dataset.relevance[query_idx])
            if score > 0
        ]
        all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        results.append({
            "query": query,
            "true_relevant": all_docs_with_scores,
            "predicted_relevant": [
                (dataset.documents[i], dataset.relevance[query_idx][i])
                for i in top_k_indices
            ],
            "precision_recall": precision_recall_scores,
            "k_values": k_values,
        })

    return results


def evaluate_ndcg(
    dataset_path: str,
    provider_name: str = "all-MiniLM-L6-v2",
    provider_type: str = "sentence_transformers",
    k_values: List[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate an embedding model using NDCG metrics.

    Args:
        dataset_path: Path to the dataset JSON file
        provider_name: Name of the model to use
        provider_type: Type of provider ("sentence_transformers" or "ollama")
        k_values: List of k values for computing metrics. If None, uses [3, 5, 10]

    Returns:
        List of dictionaries containing evaluation results for each query
    """
    dataset = SearchDataset(dataset_path)
    if dataset.evaluation_type != "ndcg":
        raise ValueError(f"Dataset {dataset_path} is not suitable for NDCG evaluation")

    if max(dataset.relevance.flatten()) <= 1:
        raise ValueError("NDCG evaluation requires graded relevance scores (>1)")

    # Number of results to retrieve is the maximum k value
    num_results = max(k_values) if k_values else 10

    # Initialize the embedding provider
    if provider_type == "sentence_transformers":
        provider = SentenceTransformersProvider(provider_name)
    elif provider_type == "ollama":
        provider = OllamaProvider(provider_name)
    elif provider_type == "cohere":
        provider = CohereProvider(provider_name)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

    # Generate embeddings
    query_embeddings = provider.encode(dataset.queries)
    doc_embeddings = provider.encode(dataset.documents)
    similarities = np.dot(query_embeddings, doc_embeddings.T)

    results = []
    for query_idx, (query, relevant_docs) in enumerate(dataset):
        top_k_indices = np.argsort(similarities[query_idx])[-num_results:][::-1]

        # Calculate NDCG scores
        actual_scores = [dataset.relevance[query_idx][i] for i in top_k_indices]
        ideal_scores = sorted(dataset.relevance[query_idx], reverse=True)
        ndcg_scores = {}
        for k in k_values:
            ndcg_scores[k] = calculate_ndcg(actual_scores, ideal_scores, k)

        # Get relevant documents with scores (same as original)
        all_docs_with_scores = [
            (doc, score)
            for doc, score in zip(dataset.documents, dataset.relevance[query_idx])
            if score > 0
        ]
        all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        results.append({
            "query": query,
            "true_relevant": all_docs_with_scores,
            "predicted_relevant": [
                (dataset.documents[i], dataset.relevance[query_idx][i])
                for i in top_k_indices
            ],
            "ndcg": ndcg_scores,
            "k_values": k_values,
        })

    return results


def evaluate_embeddings(
    dataset_path: str,
    provider_name: str = "all-MiniLM-L6-v2",
    provider_type: str = "sentence_transformers",
    k_values: List[int] = None,
) -> List[Dict[str, Any]]:
    """High-level function that delegates to the appropriate evaluation function."""
    dataset = SearchDataset(dataset_path)

    # Initialize the embedding provider
    if provider_type == "sentence_transformers":
        provider = SentenceTransformersProvider(provider_name)
    elif provider_type == "ollama":
        provider = OllamaProvider(provider_name)
    elif provider_type == "cohere":
        provider = CohereProvider(provider_name)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

    if dataset.evaluation_type == "basic":
        return evaluate_basic(dataset_path, provider_name, provider_type, k_values)
    elif dataset.evaluation_type == "ndcg":
        return evaluate_ndcg(dataset_path, provider_name, provider_type, k_values)
    else:
        raise ValueError(f"Unknown evaluation type: {dataset.evaluation_type}")
