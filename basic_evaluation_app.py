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

# Group by provider
provider_groups = defaultdict(list)
for model_name, provider_type in available_providers:
    provider_groups[provider_type].append(model_name)

# Create selection interface
st.subheader("Select Models to Evaluate")

# Initialize session state for selections if not exists
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = set()

if 'provider_expanded' not in st.session_state:
    st.session_state.provider_expanded = set()

# Create two columns: one for provider selection, one for stats
col1, col2 = st.columns([2, 1])

with col1:
    selected_models = []
    for provider_type, models in sorted(provider_groups.items()):
        # Create expander for each provider
        with st.expander(
            f"{provider_type} ({len(models)} models)",
            expanded=(provider_type in st.session_state.provider_expanded)
        ):
            # Add "Select All" checkbox for this provider with unique key
            select_all_key = f"select_all_{provider_type}"
            all_selected = st.checkbox(
                f"Select all {provider_type} models",
                key=select_all_key
            )

            st.markdown("---")  # Add separator

            # Create checkboxes for each model with unique keys
            for model in sorted(models):
                model_key = f"model_{provider_type}_{model}"
                checked = st.checkbox(
                    model,
                    value=all_selected or (model, provider_type) in st.session_state.selected_models,
                    key=model_key
                )
                if checked:
                    selected_models.append((model, provider_type))

            # Store expanded state with unique key
            expanded_key = f"keep_expanded_{provider_type}"
            if st.checkbox("Keep expanded",
                         value=provider_type in st.session_state.provider_expanded,
                         key=expanded_key):
                st.session_state.provider_expanded.add(provider_type)
            else:
                st.session_state.provider_expanded.discard(provider_type)

with col2:
    st.write("**Selection Summary**")

    # Group selected models by provider
    provider_counts = defaultdict(int)
    for _, provider_type in selected_models:
        provider_counts[provider_type] += 1

    # Display provider counts
    for provider_type, count in provider_counts.items():
        st.write(f"*{provider_type}:* {count} selected")

    total_selected = len(selected_models)
    if total_selected > 0:
        st.markdown("---")
        st.write(f"**Total models selected:** {total_selected}")
        run_eval = st.button("Run Evaluation")
    else:
        st.warning("Please select at least one model")
        run_eval = False

# Store selections in session state
st.session_state.selected_models = set(selected_models)

if run_eval:
    # Get available datasets
    dataset_paths = get_available_datasets()
    all_results = defaultdict(dict)
    # Store detailed evaluation results to avoid duplicate computation
    detailed_results = {}

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
                    # Run evaluation and store results
                    results = evaluate_embeddings(
                        dataset_path,
                        model_name,
                        provider_type,
                    )

                    # Store detailed results for reuse
                    key = (model_name, provider_type, dataset_name)
                    detailed_results[key] = results

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
                metric_type, k_value = metric.split('@')
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

            # Use the stored results instead of re-evaluating
            key = (model_name, provider_type, dataset_name)
            if key in detailed_results:
                results = detailed_results[key]

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

                        # Metrics display
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
            else:
                st.error(f"No results available for {model_name} ({provider_type}) on {dataset_name}")

            st.markdown("---")  # Add separator between datasets
