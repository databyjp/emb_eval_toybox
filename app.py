import os
import streamlit as st
import pandas as pd
from emb_eval_toybox.evaluation import evaluate_embeddings
from emb_eval_toybox.registry import get_available_providers, get_available_datasets
from emb_eval_toybox.data.dataset import SearchDataset
import torch

torch.classes.__path__ = []


def load_dataset_preview(dataset_path, num_rows=5):
    """Load and return a preview of the dataset."""
    # Note: You might need to adjust this based on your actual dataset format
    try:
        df = pd.read_json(dataset_path)  # or json, depending on your format
        return df.head(num_rows)
    except Exception as e:
        return f"Error loading dataset: {str(e)}"


def display_dataset_overview(dataset: SearchDataset):
    """Display overview information about a dataset."""
    with st.expander("Dataset Information", expanded=True):
        st.write(f"**Name:** {dataset.name}")
        st.write(f"**Evaluation Type:** {dataset.evaluation_type}")
        if dataset.evaluation_type == "basic":
            st.write("_(Uses precision/recall metrics)_")
        st.write(f"**Description:** {dataset.description}")

        # Dataset statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Queries", len(dataset.queries))
        with col2:
            st.metric("Documents", len(dataset.documents))
        with col3:
            relevance_levels = sorted(set(dataset.relevance.flatten()))
            st.metric("Relevance Levels", f"{min(relevance_levels)}-{max(relevance_levels)}")

    # Sample content with fixed heights
    st.subheader("Sample Content")

    # Show queries with fixed height
    with st.expander("Queries"):
        query_df = pd.DataFrame({
            "Query": dataset.queries,
            "Relevant Documents": [
                len([r for r in dataset.relevance[i] if r > 0])
                for i in range(len(dataset.queries))
            ]
        })
        st.dataframe(query_df, height=300)  # Fixed height

    # Show documents with fixed height
    with st.expander("Documents"):
        doc_df = pd.DataFrame({
            "Document": dataset.documents,
            "Times Referenced": [
                len([r for r in dataset.relevance[:, i] if r > 0])
                for i in range(len(dataset.documents))
            ]
        })
        st.dataframe(doc_df, height=300)  # Fixed height

    # Show relevance matrix with fixed height
    with st.expander("Relevance Matrix"):
        relevance_df = pd.DataFrame(
            dataset.relevance,
            columns=[f"Doc {i}" for i in range(len(dataset.documents))],
            index=[f"Query {i}" for i in range(len(dataset.queries))]
        )
        st.dataframe(relevance_df, height=300)  # Fixed height


def main():
    st.title("Embedding Model Evaluation")

    # Create tabs
    overview_tab, evaluation_tab = st.tabs(["Dataset Overview", "Model Evaluation"])

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Dataset selection
    available_datasets = get_available_datasets()
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        list(available_datasets.keys()),
    )
    dataset_path = available_datasets[selected_dataset]
    dataset = SearchDataset(dataset_path)

    with overview_tab:
        display_dataset_overview(dataset)

    with evaluation_tab:
        # Model selection
        available_models = get_available_providers()
        selected_models = st.multiselect(
            "Select Models to Evaluate",
            [f"{model} ({provider})" for model, provider in available_models],
            default=[f"{available_models[0][0]} ({available_models[0][1]})"],
        )

        # K-values selection
        k_values = st.multiselect(
            "Select k values for evaluation", [1, 3, 5, 10], default=[3]
        )

        # Evaluation section
        st.header("Evaluation Metrics")
        st.write("The following metrics will be calculated:")
        if dataset.evaluation_type == "basic":
            st.write("- Precision@k")
            st.write("- Recall@k")
        else:  # ndcg
            st.write("- NDCG@k")

        # Run evaluation button
        if st.button("Run Evaluation"):
            for model_spec in selected_models:
                model_name = model_spec.split(" (")[0]
                provider_type = model_spec.split("(")[1].rstrip(")")

                try:
                    results = evaluate_embeddings(
                        dataset_path,
                        model_name,
                        provider_type,
                        k_values=k_values
                    )
                    st.subheader(f"Results for {model_name}")
                    for result in results:
                        with st.expander(f"Query: {result['query']}"):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("Predicted Relevant Documents:")
                                for doc, _ in result["predicted_relevant"]:
                                    st.write(f"- {doc}")

                            with col2:
                                st.write("True Relevant Documents:")
                                for doc, _ in result["true_relevant"]:
                                    st.write(f"- {doc}")

                            st.write("Metrics:")
                            for k in result["precision_recall"].keys():
                                pr_scores = result["precision_recall"][k]
                                st.write(f"At k={k}:")
                                st.write(f"- Precision: {pr_scores['precision']:.3f}")
                                st.write(f"- Recall: {pr_scores['recall']:.3f}")

                except Exception as e:
                    st.error(f"Error evaluating {model_name}: {str(e)}")


if __name__ == "__main__":
    main()
