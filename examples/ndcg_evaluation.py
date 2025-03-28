from pathlib import Path
import numpy as np
from emb_eval_toybox.data.dataset import SearchDataset
from emb_eval_toybox.providers import SentenceTransformersProvider, OllamaProvider


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

    # Convert relevance scores to gains (2^relevance - 1)
    gains = [2**score - 1 for score in relevance_scores]

    # Calculate DCG
    dcg = gains[0]  # First result
    for i, gain in enumerate(gains[1:], 1):
        dcg += gain / np.log2(i + 1)  # +1 because log2(1) is 0

    return dcg


def calculate_ndcg(relevance_scores: list[int], ideal_scores: list[int], k: int = None) -> float:
    """Calculate Normalized Discounted Cumulative Gain.

    Args:
        relevance_scores: List of actual relevance scores
        ideal_scores: List of ideal relevance scores (sorted by relevance)
        k: Number of results to consider. If None, uses all results.

    Returns:
        NDCG score between 0 and 1
    """
    dcg = calculate_dcg(relevance_scores, k)
    idcg = calculate_dcg(ideal_scores, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_embeddings(
    dataset_path: str,
    provider_name: str = "all-MiniLM-L6-v2",
    provider_type: str = "sentence_transformers",
    num_results: int = 5,
):
    """Evaluate an embedding model on a given dataset using NDCG.

    Args:
        dataset_path: Path to the dataset JSON file
        provider_name: Name of the model to use
        provider_type: Type of provider ("sentence_transformers" or "ollama")
        num_results: Number of results to retrieve for each query
    """
    # Load the dataset
    dataset = SearchDataset(dataset_path)

    # Initialize the embedding provider
    if provider_type == "sentence_transformers":
        provider = SentenceTransformersProvider(provider_name)
    elif provider_type == "ollama":
        provider = OllamaProvider(provider_name)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

    print(f"Using model: {provider.name}")
    print(f"Embedding dimension: {provider.dimension}")

    # Generate embeddings for queries and documents
    query_embeddings = provider.encode(dataset.queries)
    doc_embeddings = provider.encode(dataset.documents)

    # Calculate cosine similarity between queries and documents
    similarities = np.dot(query_embeddings, doc_embeddings.T)

    # Calculate metrics
    results = []
    for query_idx, (query, relevant_docs) in enumerate(dataset):
        # Get top-k results for this query
        top_k_indices = np.argsort(similarities[query_idx])[-num_results:][::-1]

        # Get actual relevance scores for predicted order
        actual_scores = [dataset.relevance[query_idx][i] for i in top_k_indices]

        # Get ideal relevance scores (sorted by relevance)
        ideal_scores = sorted(dataset.relevance[query_idx], reverse=True)

        # Calculate NDCG at different k values
        ndcg_scores = {}
        for k in [3, 5, 10]:  # Calculate NDCG at different positions
            ndcg_scores[k] = calculate_ndcg(actual_scores, ideal_scores, k)

        # Get all documents with their relevance scores
        all_docs_with_scores = [
            (doc, score) for doc, score in zip(dataset.documents, dataset.relevance[query_idx])
            if score > 0  # Only include documents with non-zero relevance
        ]
        # Sort by relevance score
        all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        results.append(
            {
                "query": query,
                "true_relevant": all_docs_with_scores,
                "predicted_relevant": [(dataset.documents[i], dataset.relevance[query_idx][i]) for i in top_k_indices],
                "ndcg": ndcg_scores,
                "num_results": num_results,
            }
        )

    return results


def print_results(results):
    """Print evaluation results."""
    print("\nEvaluation Results:")
    print("-" * 50)
    for result in results:
        print(f"\nQuery: {result['query']}")
        print(f"\nTrue Relevant ({len(result['true_relevant'])}):")
        for doc, score in result["true_relevant"]:
            print(f"  - [Score: {score}] {doc}")
        print(f"\nPredicted Relevant ({len(result['predicted_relevant'])}):")
        for doc, score in result["predicted_relevant"]:
            print(f"  - [Score: {score}] {doc}")
        print("\n=== Evaluation Metrics ===")
        for k, score in result["ndcg"].items():
            print(f"NDCG@{k}: {score:.3f}")
        print("-" * 50)


def evaluate_provider(dataset_path: str, provider_name: str, provider_type: str):
    """Evaluate a single provider and print results."""
    print(
        f"\nEvaluating with {provider_type} ({provider_name}) on dataset: {dataset_path}"
    )
    results = evaluate_embeddings(dataset_path, provider_name, provider_type)
    print_results(results)
    return results


if __name__ == "__main__":
    # Define the dataset and providers to evaluate
    dataset_path = "data/synthetic_dataset_trivia_graded.json"

    # List of providers to evaluate
    providers = [
        ("all-MiniLM-L6-v2", "sentence_transformers"),
        ("snowflake-arctic-embed:22m", "ollama"),
        ("snowflake-arctic-embed2", "ollama"),
    ]

    # Evaluate each provider
    for model_name, provider_type in providers:
        try:
            evaluate_provider(dataset_path, model_name, provider_type)
        except Exception as e:
            print(f"Error evaluating {provider_type} ({model_name}): {str(e)}")
            continue
