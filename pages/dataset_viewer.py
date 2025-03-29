import streamlit as st
import json
from pathlib import Path
from emb_eval_toybox.registry import get_available_datasets
from emb_eval_toybox.data.dataset import SearchDataset
import pandas as pd

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

    # Display queries and their relevant documents
    st.subheader("Queries and Documents")

    for query_idx, (query, _) in enumerate(dataset):
        with st.expander(f"Query: {query}"):
            # Get all documents with their relevance scores
            documents_with_scores = list(zip(dataset.documents, dataset.relevance[query_idx]))

            # Split into relevant and irrelevant
            relevant_docs = [(doc, score) for doc, score in documents_with_scores if score > 0]
            irrelevant_docs = [(doc, score) for doc, score in documents_with_scores if score == 0]

            # Sort both lists by score (descending)
            relevant_docs.sort(key=lambda x: x[1], reverse=True)
            irrelevant_docs.sort(key=lambda x: x[1], reverse=True)

            # Display relevant documents
            st.write("**Relevant Documents:**")
            if relevant_docs:
                df_relevant = pd.DataFrame(relevant_docs, columns=['Document', 'Relevance Score'])
                st.dataframe(
                    df_relevant.style.format({'Relevance Score': '{:.2f}'}),
                    use_container_width=True
                )
            else:
                st.write("*No relevant documents*")

            # Display irrelevant documents
            st.write("**Irrelevant Documents:**")
            if irrelevant_docs:
                df_irrelevant = pd.DataFrame(irrelevant_docs, columns=['Document', 'Relevance Score'])
                st.dataframe(
                    df_irrelevant.style.format({'Relevance Score': '{:.2f}'}),
                    use_container_width=True
                )
            else:
                st.write("*No irrelevant documents*")

            # Display statistics
            st.write(f"**Number of relevant documents:** {len(relevant_docs)}")

    # Display overall dataset statistics
    st.subheader("Dataset Statistics")
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

    # Dataset metadata in collapsible section at the bottom
    with st.expander("Dataset Metadata"):
        st.json(dataset.metadata)
