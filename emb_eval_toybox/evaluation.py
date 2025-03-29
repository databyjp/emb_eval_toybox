"""Core evaluation functionality for embedding models."""

from pathlib import Path
import numpy as np
from typing import List, Dict, Any
from .data.dataset import SearchDataset
from .providers import create_provider
from .providers.base import EmbeddingProvider


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
    dcg = 0
    for i, gain in enumerate(gains):
        # Position 1 -> log_2(2) = 1
        # Position 2 -> log_2(3)
        # Position 3 -> log_2(4)
        dcg += gain / np.log2(i + 2) if i > 0 else gain

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
    dataset: SearchDataset,
    provider: EmbeddingProvider,
    k_values: List[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate an embedding model using basic metrics (precision/recall).

    Args:
        dataset: SearchDataset object
        provider: EmbeddingProvider object
        k_values: List of k values for computing metrics. If None, uses [3, 5, 10]

    Returns:
        List of dictionaries containing evaluation results for each query
    """
    if k_values is None:
        k_values = [3, 5, 10]

    # Generate embeddings
    query_embeddings = provider.encode(dataset.queries)
    doc_embeddings = provider.encode(dataset.documents)
    similarities = np.dot(query_embeddings, doc_embeddings.T)

    results = []
    for query_idx, (query, relevant_docs) in enumerate(dataset):
        top_k_indices = np.argsort(similarities[query_idx])[-max(k_values):][::-1]

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
    dataset: SearchDataset,
    provider: EmbeddingProvider,
    k_values: List[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate an embedding model using NDCG metrics.

    Args:
        dataset: SearchDataset object
        provider: EmbeddingProvider object
        k_values: List of k values for computing metrics. If None, uses [3, 5, 10]

    Returns:
        List of dictionaries containing evaluation results for each query
    """
    if k_values is None:
        k_values = [3, 5, 10]

    # Generate embeddings
    query_embeddings = provider.encode(dataset.queries)
    doc_embeddings = provider.encode(dataset.documents)
    similarities = np.dot(query_embeddings, doc_embeddings.T)

    results = []
    for query_idx, (query, relevant_docs) in enumerate(dataset):
        # Get full ranking from similarities
        predicted_ranking = np.argsort(similarities[query_idx])[::-1]  # Sort in descending order

        # Calculate NDCG scores
        ndcg_scores = {}
        for k in k_values:
            # Get top k predictions and their true relevance scores
            top_k_indices = predicted_ranking[:k]
            predicted_relevance = [dataset.relevance[query_idx][i] for i in top_k_indices]

            # Get ideal relevance scores
            ideal_relevance = sorted(dataset.relevance[query_idx], reverse=True)[:k]

            ndcg_scores[k] = calculate_ndcg(predicted_relevance, ideal_relevance, k)

        # Get predicted documents with their similarity scores
        predicted_docs = [
            (dataset.documents[i], similarities[query_idx][i])
            for i in predicted_ranking[:max(k_values)]
        ]

        # Get true relevant documents (unchanged)
        true_relevant = [
            (doc, score)
            for doc, score in zip(dataset.documents, dataset.relevance[query_idx])
            if score > 0
        ]
        true_relevant.sort(key=lambda x: x[1], reverse=True)

        results.append({
            "query": query,
            "true_relevant": true_relevant,
            "predicted_relevant": predicted_docs,
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

    # Initialize the embedding provider using the factory function
    provider = create_provider(provider_type, provider_name)

    if dataset.evaluation_type == "basic":
        return evaluate_basic(dataset, provider, k_values)
    elif dataset.evaluation_type == "ndcg":
        return evaluate_ndcg(dataset, provider, k_values)
    else:
        raise ValueError(f"Unknown evaluation type: {dataset.evaluation_type}")
