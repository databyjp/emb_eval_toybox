"""Example script for NDCG evaluation of embedding models."""

from emb_eval_toybox.evaluation import evaluate_embeddings
from emb_eval_toybox.registry import get_available_providers, get_available_datasets

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

def main():
    """Run NDCG evaluation on all available providers."""
    dataset_path = get_available_datasets()["Trivia (Graded)"]

    for model_name, provider_type in get_available_providers():
        try:
            print(f"\nEvaluating with {provider_type} ({model_name}) on dataset: {dataset_path}")
            results = evaluate_embeddings(
                dataset_path,
                model_name,
                provider_type,
                k_values=[1, 3]  # Show metrics at multiple k values
            )
            print_results(results)
        except Exception as e:
            print(f"Error evaluating {provider_type} ({model_name}): {str(e)}")
            continue

if __name__ == "__main__":
    main()
