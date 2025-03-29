"""Example script for basic evaluation of embedding models."""

from emb_eval_toybox.evaluation import evaluate_embeddings
from emb_eval_toybox.registry import get_available_providers, get_available_datasets
from collections import defaultdict


def print_results(results):
    """Print evaluation results in a standardized format."""
    print("\nEvaluation Results:")
    print("-" * 50)
    for result in results:
        print(f"\nQuery: {result['query']}")

        print(f"\nTrue Relevant ({len(result['true_relevant'])}):")
        for doc, score in result["true_relevant"]:
            print(f"  - [Score: {score:.3f}] {doc}")

        print(f"\nPredicted Relevant ({len(result['predicted_relevant'])}):")
        for doc, score in result["predicted_relevant"]:
            print(f"  - [Score: {score:.3f}] {doc}")

        print("\n=== Evaluation Metrics ===")
        # Handle NDCG metrics if present
        if "ndcg" in result:
            for k, score in result["ndcg"].items():
                print(f"NDCG@{k}: {score:.3f}")

        # Handle precision/recall metrics if present
        if "precision_recall" in result:
            for k, pr_scores in result["precision_recall"].items():
                print(f"At k={k}:")
                print(f"  Precision: {pr_scores['precision']:.3f}")
                print(f"  Recall: {pr_scores['recall']:.3f}")
        print("-" * 50)


def print_model_summary(results_by_model):
    """Print a summary table comparing all evaluated models."""
    # Determine all unique metrics across all results
    all_metrics = set()
    for (model, provider), model_results in results_by_model.items():
        for metric_dict in model_results.values():
            all_metrics.update(metric_dict.keys())
    metrics_list = sorted(all_metrics)

    # Calculate column widths
    model_width = max(len("Model"), max(len(model) for model, _ in results_by_model.keys()))
    provider_width = max(len("Provider"), max(len(provider) for _, provider in results_by_model.keys()))
    metric_width = max(len(metric) for metric in metrics_list)

    # Print header
    print("\n=== Model Comparison Summary ===")
    header = (
        f"{'Model':<{model_width}} | "
        f"{'Provider':<{provider_width}} | "
        f"{' | '.join(metric.ljust(metric_width) for metric in metrics_list)}"
    )
    print(header)
    print("-" * len(header))

    # Print each model's results
    for (model, provider), model_results in results_by_model.items():
        metrics_avg = defaultdict(list)
        for metric_dict in model_results.values():
            for metric, value in metric_dict.items():
                metrics_avg[metric].append(value)

        # Calculate averages and format row
        metric_values = []
        for metric in metrics_list:
            values = metrics_avg.get(metric, [0])
            avg = sum(values) / len(values) if values else 0
            metric_values.append(f"{avg:.3f}".ljust(metric_width))

        row = (
            f"{model:<{model_width}} | "
            f"{provider:<{provider_width}} | "
            f"{' | '.join(metric_values)}"
        )
        print(row)
    print("-" * len(header))


def main():
    """Run basic evaluation on all available providers."""
    dataset_paths = get_available_datasets()
    # Dictionary to store results for all models
    all_results = defaultdict(dict)

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
                    k_values=[1],
                )
                print_results(results)

                # Collect metrics for summary
                metrics = {}
                for result in results:
                    if "ndcg" in result:
                        for k, score in result["ndcg"].items():
                            metrics[f"NDCG@{k}"] = score
                    if "precision_recall" in result:
                        for k, pr_scores in result["precision_recall"].items():
                            metrics[f"Precision@{k}"] = pr_scores["precision"]
                            metrics[f"Recall@{k}"] = pr_scores["recall"]

                all_results[(model_name, provider_type)][dataset_name] = metrics

            except Exception as e:
                print(f"Error evaluating {provider_type} ({model_name}): {str(e)}")
                continue

    # Print final summary table
    print_model_summary(all_results)


if __name__ == "__main__":
    main()
