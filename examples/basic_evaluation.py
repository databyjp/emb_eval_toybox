"""Example script for basic evaluation of embedding models."""

from emb_eval_toybox.evaluation import evaluate_embeddings
from emb_eval_toybox.registry import get_available_providers, get_available_datasets


def print_results(results):
    """Print basic evaluation results."""
    print("\nEvaluation Results:")
    print("-" * 50)
    for result in results:
        print(f"\nQuery: {result['query']}")

        print("\nPredicted Relevant Documents:")
        for doc, score in result["predicted_relevant"]:
            print(f"  - {doc}")

        print("\nTrue Relevant Documents:")
        for doc, score in result["true_relevant"]:
            print(f"  - {doc}")

        print("\nMetrics:")
        for k in result["precision_recall"].keys():
            pr_scores = result["precision_recall"][k]
            print(f"  At k={k}:")
            print(f"    Precision: {pr_scores['precision']:.3f}")
            print(f"    Recall: {pr_scores['recall']:.3f}")

        print("-" * 50)


def main():
    """Run basic evaluation on all available providers."""
    dataset_paths = get_available_datasets()
    for dataset_name, dataset_path in dataset_paths.items():
        print(f"Evaluating on dataset: {dataset_name}")
        for model_name, provider_type in get_available_providers():
            try:
                print(
                    f"\nEvaluating with {provider_type} ({model_name}) on dataset: {dataset_path}"
                )
                results = evaluate_embeddings(
                    dataset_path,
                    model_name,
                    provider_type,
                    k_values=[3],  # Only show basic metrics at k=3
                )
                print_results(results)
            except Exception as e:
                print(f"Error evaluating {provider_type} ({model_name}): {str(e)}")
                continue


if __name__ == "__main__":
    main()
