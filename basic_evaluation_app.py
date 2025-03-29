import streamlit as st
import pandas as pd
import plotly.express as px
from emb_eval_toybox.registry import get_available_providers, get_available_datasets
from emb_eval_toybox.evaluation import evaluate_embeddings
from collections import defaultdict
import torch

torch.classes.__path__ = []

st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")

st.title("Model Evaluation Dashboard")
st.write("Use this page to evaluate and compare different embedding models. To explore the available datasets, use the Dataset Explorer page in the sidebar.")

# Get available models and providers
available_providers = list(get_available_providers())

# Group and sort by provider
provider_groups = defaultdict(list)
for model_name, provider_type in available_providers:
    provider_groups[provider_type].append(model_name)

# Sort providers and models within each provider
sorted_options = []
for provider_type in sorted(provider_groups.keys()):
    for model_name in sorted(provider_groups[provider_type]):
        sorted_options.append((model_name, provider_type))

# Model selection with formatted display
selected_models = st.multiselect(
    "Select models to evaluate",
    options=sorted_options,
    default=[sorted_options[0]] if sorted_options else None,
    format_func=lambda x: f"{x[1]}: {x[0]}"
)

if st.button("Run Evaluation"):
    if not selected_models:
        st.warning("Please select at least one model to evaluate.")
    else:
        # Get available datasets
        dataset_paths = get_available_datasets()
        all_results = defaultdict(dict)

        # Create summary progress bar
        progress_bar = st.progress(0)
        total_evaluations = len(selected_models) * len(dataset_paths)
        current_evaluation = 0

        for model_name, provider_type in selected_models:
            for dataset_name, dataset_path in dataset_paths.items():
                with st.spinner(
                    f"Evaluating {model_name} ({provider_type}) on {dataset_name}..."
                ):
                    try:
                        results = evaluate_embeddings(
                            dataset_path,
                            model_name,
                            provider_type,
                        )

                        # Collect and average metrics
                        metrics_accumulator = defaultdict(list)
                        for result in results:
                            if "ndcg" in result:
                                for k, score in result["ndcg"].items():
                                    metrics_accumulator[f"NDCG@{k}"].append(score)
                            if "precision_recall" in result:
                                for k, pr_scores in result["precision_recall"].items():
                                    metrics_accumulator[f"Precision@{k}"].append(
                                        pr_scores["precision"]
                                    )
                                    metrics_accumulator[f"Recall@{k}"].append(
                                        pr_scores["recall"]
                                    )

                        # Average the accumulated metrics
                        averaged_metrics = {
                            metric: sum(scores) / len(scores)
                            for metric, scores in metrics_accumulator.items()
                        }

                        all_results[(model_name, provider_type)][
                            dataset_name
                        ] = averaged_metrics

                    except Exception as e:
                        st.error(
                            f"Error evaluating {provider_type} ({model_name}): {str(e)}"
                        )
                        continue

                current_evaluation += 1
                progress_bar.progress(current_evaluation / total_evaluations)

        # Group metrics by type and k value
        metric_groups = defaultdict(lambda: defaultdict(list))
        for (model, provider), dataset_results in all_results.items():
            for dataset, metrics in dataset_results.items():
                for metric, value in metrics.items():
                    metric_type = metric.split('@')[0]  # NDCG, Precision, or Recall
                    k_value = metric.split('@')[1]      # The k value
                    metric_groups[metric_type][k_value].append({
                        "Model": model,
                        "Provider": provider,
                        "Dataset": dataset,
                        "Value": value
                    })

        # Display results for each metric type and k value
        metric_types = ["NDCG", "Precision", "Recall"]
        tabs = st.tabs(metric_types)

        for metric_type, tab in zip(metric_types, tabs):
            with tab:
                # Get all k values for this metric type
                k_values = sorted(metric_groups[metric_type].keys())

                for k in k_values:
                    st.subheader(f"{metric_type}@{k}")

                    # Create dataframe for this specific metric and k value
                    df = pd.DataFrame(metric_groups[metric_type][k])
                    st.dataframe(df)

                    # Create visualization
                    fig = px.bar(df,
                               x="Model",
                               y="Value",
                               color="Dataset",
                               title=f"{metric_type}@{k} Comparison",
                               barmode="group")
                    st.plotly_chart(fig)

        # Detailed results with flattened expanders
        st.subheader("Detailed Results")

        for model_name, provider_type in selected_models:
            st.subheader(f"{model_name} ({provider_type})")

            for dataset_name, dataset_path in dataset_paths.items():
                st.write(f"**Dataset: {dataset_name}**")
                results = evaluate_embeddings(dataset_path, model_name, provider_type)

                for i, result in enumerate(results):
                    with st.expander(f"Query: {result['query']}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**True Relevant Documents:**")
                            for doc, score in result["true_relevant"]:
                                st.write(f"- [{score:.3f}] {doc}")

                        with col2:
                            st.write("**Predicted Relevant Documents:**")
                            for doc, score in result["predicted_relevant"]:
                                st.write(f"- [{score:.3f}] {doc}")

                        st.write("**Metrics:**")
                        metrics_col1, metrics_col2 = st.columns(2)

                        with metrics_col1:
                            if "ndcg" in result:
                                st.write("NDCG Scores:")
                                for k, score in result["ndcg"].items():
                                    st.write(f"- NDCG@{k}: {score:.3f}")

                        with metrics_col2:
                            if "precision_recall" in result:
                                st.write("Precision/Recall Scores:")
                                for k, pr_scores in result["precision_recall"].items():
                                    st.write(f"- At k={k}:")
                                    st.write(f"  Precision: {pr_scores['precision']:.3f}")
                                    st.write(f"  Recall: {pr_scores['recall']:.3f}")
                st.markdown("---")  # Add separator between datasets
