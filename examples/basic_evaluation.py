from pathlib import Path
import numpy as np
from emb_eval_toybox.data.dataset import SearchDataset
from emb_eval_toybox.providers import SentenceTransformersProvider, OllamaProvider


def evaluate_embeddings(
    dataset_path: str,
    provider_name: str = "all-MiniLM-L6-v2",
    provider_type: str = "sentence_transformers",
):
    """Evaluate an embedding model on a given dataset.

    Args:
        dataset_path: Path to the dataset JSON file
        provider_name: Name of the model to use
        provider_type: Type of provider ("sentence_transformers" or "ollama")
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
        # Get number of relevant documents for this query
        k = len(relevant_docs)

        # Get top-k results for this query
        top_k_indices = np.argsort(similarities[query_idx])[-k:][::-1]

        # Calculate precision@k
        true_positives = len(set(top_k_indices) & set(relevant_docs))
        precision = true_positives / k

        # Calculate recall@k
        recall = true_positives / len(relevant_docs)

        results.append(
            {
                "query": query,
                "true_relevant": [dataset.documents[i] for i in relevant_docs],
                "predicted_relevant": [dataset.documents[i] for i in top_k_indices],
                "precision@k": precision,
                "recall@k": recall,
                "k": k,
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
        for doc in result["true_relevant"]:
            print(f"  - {doc}")
        print(f"\nPredicted Relevant ({len(result['predicted_relevant'])}):")
        for doc in result["predicted_relevant"]:
            print(f"  - {doc}")
        print("\n=== Evaluation Metrics ===")
        print(f"Precision@{result['k']}: {result['precision@k']:.2f}")
        print(f"Recall@{result['k']}: {result['recall@k']:.2f}")
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
    dataset_path = "data/synthetic_dataset_instruments.json"

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
