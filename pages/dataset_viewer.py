import streamlit as st
import json
from pathlib import Path
from emb_eval_toybox.registry import get_available_datasets
from emb_eval_toybox.data.dataset import SearchDataset
import pandas as pd
from emb_eval_toybox.visualization import display_document_card

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("Dataset Explorer")

# Get available datasets
dataset_paths = get_available_datasets()
selected_dataset = st.selectbox(
    "Select Dataset",
    options=list(dataset_paths.keys()),
    format_func=lambda x: x
)

if selected_dataset:
    # Load the dataset
    dataset = SearchDataset(dataset_paths[selected_dataset])

    # Display dataset metadata
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Queries", len(dataset.queries))
    with col2:
        st.metric("Number of Documents", len(dataset.documents))
    with col3:
        avg_relevant = sum(
            sum(1 for score in query_scores if score > 0)
            for query_scores in dataset.relevance
        ) / len(dataset.queries)
        st.metric("Average Relevant Docs per Query", f"{avg_relevant:.1f}")

    with st.expander("Dataset Metadata"):
        st.json(dataset.metadata)

    # Display queries and their relevant documents
    st.subheader("Queries and Documents")

    for query_idx, (query, _) in enumerate(dataset):
        query_id = dataset.get_query_id(query_idx)
        # Truncate query text for expander header (show first 50 chars + "..." if longer)
        truncated_query = (query[:50] + "...") if len(query) > 50 else query

        with st.expander(f"Query {query_id}: {truncated_query}"):
            # Display full query in larger font
            st.markdown(f"### {query}")
            st.markdown("---")  # Add separator

            # Get all documents with their relevance scores and explanations
            documents_data = []
            for doc_idx, score in enumerate(dataset.relevance[query_idx]):
                doc_data = {
                    "id": dataset.get_document_id(doc_idx),
                    "text": dataset.documents[doc_idx],
                    "score": score,
                }

                # Add explanation if available
                explanation = dataset.get_explanation(query_idx, doc_idx)
                if explanation:
                    doc_data["explanation"] = explanation

                documents_data.append(doc_data)

            # Add sorting options
            sort_by = st.radio(
                "Sort documents by:",
                ["Relevance (high to low)", "Document ID"],
                key=f"sort_{query_id}",
                horizontal=True
            )

            # Sort documents according to selection
            if sort_by == "Relevance (high to low)":
                documents_data.sort(key=lambda x: (x["score"], x["id"]), reverse=True)
            else:  # Document ID
                documents_data.sort(key=lambda x: x["id"])

            # Display all documents
            st.write(f"**Documents (sorted by {sort_by.lower()}):**")
            for doc in documents_data:
                display_document_card(
                    doc["id"],
                    doc["text"],
                    doc["score"],
                    doc.get("explanation"),
                    is_true_document=True
                )
