"""Example script for NDCG evaluation of embedding models."""

from emb_eval_toybox.evaluation import evaluate_embeddings
from emb_eval_toybox.registry import get_available_providers, get_available_datasets
from emb_eval_toybox.data.dataset import SearchDataset
from collections import defaultdict
from basic_evaluation import print_results, print_model_summary

def main():
    """Run NDCG evaluation on all available providers."""
    dataset_path = get_available_datasets()["Trivia (Graded)"]
    dataset = SearchDataset(dataset_path)

    if dataset.evaluation_type != "ndcg":
        raise ValueError(
            f"Dataset {dataset_path} is not suitable for NDCG evaluation. "
            f"It is marked for {dataset.evaluation_type} evaluation."
        )

    print(f"Using dataset: {dataset.description}")

    # Dictionary to store results for all models
    all_results = defaultdict(dict)

    for model_name, provider_type in get_available_providers():
        try:
            print(
                f"\nEvaluating with {provider_type} ({model_name}) on dataset: {dataset_path}"
            )
            results = evaluate_embeddings(
                dataset_path,
                model_name,
                provider_type,
                k_values=[3],  # Show metrics at multiple k values
            )
            print_results(results)

            # Collect metrics for summary
            metrics = {}
            for result in results:
                for k, score in result["ndcg"].items():
                    metrics[f"NDCG@{k}"] = score

            all_results[(model_name, provider_type)]["Trivia (Graded)"] = metrics

        except Exception as e:
            print(f"Error evaluating {provider_type} ({model_name}): {str(e)}")
            continue

    # Print final summary table
    print_model_summary(all_results)

if __name__ == "__main__":
    main()
