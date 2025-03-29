import os
import streamlit as st
import pandas as pd
from emb_eval_toybox.evaluation import evaluate_embeddings
from emb_eval_toybox.registry import get_available_providers, get_available_datasets
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


def main():
    st.title("Embedding Model Evaluation")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Dataset selection
    available_datasets = get_available_datasets()
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        list(available_datasets.keys()),
    )
    dataset_path = available_datasets[selected_dataset]

    # Model selection
    available_models = get_available_providers()
    selected_models = st.sidebar.multiselect(
        "Select Models to Evaluate",
        [f"{model} ({provider})" for model, provider in available_models],
        default=[f"{available_models[0][0]} ({available_models[0][1]})"],
    )

    # K-values selection
    k_values = st.sidebar.multiselect(
        "Select k values for evaluation", [1, 3, 5, 10], default=[3]
    )

    # Main content area
    st.header("Dataset Preview")
    st.write(f"Selected dataset: {selected_dataset}")
    # preview = load_dataset_preview(dataset_path)
    # st.dataframe(preview)

    st.header("Evaluation Metrics")
    st.write("The following metrics will be calculated:")
    st.write("- Precision@k")
    st.write("- Recall@k")

    # Run evaluation
    if st.button("Start Evaluation"):
        if not selected_models:
            st.error("Please select at least one model to evaluate.")
            return

        st.write("Running evaluation...")
        progress_bar = st.progress(0)

        for i, model_full_name in enumerate(selected_models):
            model_name = model_full_name.split(" (")[0]
            provider_type = model_full_name.split("(")[1].rstrip(")")

            try:
                results = evaluate_embeddings(
                    dataset_path, model_name, provider_type, k_values=k_values
                )

                # Display results
                st.subheader(f"Results for {model_full_name}")
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
                st.error(f"Error evaluating {model_full_name}: {str(e)}")

            progress_bar.progress((i + 1) / len(selected_models))

        st.success("Evaluation complete!")


if __name__ == "__main__":
    main()
