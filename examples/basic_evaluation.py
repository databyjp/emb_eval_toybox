from pathlib import Path
import numpy as np
from emb_eval_toybox.data.dataset import SearchDataset
from emb_eval_toybox.providers import SentenceTransformersProvider


def evaluate_embeddings(provider_name: str = "all-MiniLM-L6-v2"):
    """Evaluate a sentence transformer model on our coffee dataset."""
    # Load the dataset
    dataset = SearchDataset("data/synthetic_dataset_coffee.json")

    # Initialize the embedding provider
    provider = SentenceTransformersProvider(provider_name)
    print(f"Using provider: {provider.name}")
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
                "k": k
            }
        )

    return results


if __name__ == "__main__":
    # Run evaluation
    results = evaluate_embeddings()

    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for result in results:
        print(f"\nQuery: {result['query']}")
        print(f"True Relevant ({len(result['true_relevant'])}):")
        for doc in result['true_relevant']:
            print(f"  - {doc}")
        print(f"Predicted Relevant ({len(result['predicted_relevant'])}):")
        for doc in result['predicted_relevant']:
            print(f"  - {doc}")
        print(f"Precision@{result['k']}: {result['precision@k']:.2f}")
        print(f"Recall@{result['k']}: {result['recall@k']:.2f}")
        print("-" * 50)
