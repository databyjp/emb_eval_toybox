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

    # Calculate maximum number of relevant documents
    max_relevant = max(
        sum(1 for score in query_scores if score > 0)
        for query_scores in dataset.relevance
    )

    # Choose appropriate k values for NDCG
    suggested_k = [3, 5, 10]
    k_values = [k for k in suggested_k if k <= max(max_relevant * 3, 10)]

    print(f"Using dataset: {dataset.description}")
    print(f"Evaluating at k values: {k_values} (based on {max_relevant} maximum relevant documents)")

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
                k_values=k_values,
            )
            print_results(results)

            # Collect metrics and average across queries
            metrics_accumulator = defaultdict(list)
            for result in results:
                for k, score in result["ndcg"].items():
                    metrics_accumulator[f"NDCG@{k}"].append(score)

            # Calculate averages
            averaged_metrics = {
                metric: sum(scores) / len(scores)
                for metric, scores in metrics_accumulator.items()
            }

            all_results[(model_name, provider_type)]["Trivia (Graded)"] = averaged_metrics

        except Exception as e:
            print(f"Error evaluating {provider_type} ({model_name}): {str(e)}")
            continue

    # Print final summary table
    print_model_summary(all_results)

if __name__ == "__main__":
    main()
