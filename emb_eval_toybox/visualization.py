"""Reusable visualization components for Streamlit apps."""

import streamlit as st
from typing import Optional, Dict, Any


def display_document_card(
    doc_id: str,
    doc_text: str,
    score: float,
    explanation: Optional[str] = None,
    true_relevance: Optional[int] = None,
    is_true_document: bool = False
) -> None:
    """Display a document card with relevance information.

    Args:
        doc_id: Document identifier
        doc_text: Document text content
        score: Relevance score (int) for true documents or similarity score (float) for predictions
        explanation: Optional explanation of relevance
        true_relevance: Optional true relevance score (for predicted documents)
        is_true_document: Whether this is a true relevant document vs a prediction
    """
    with st.container():
        # Use columns for layout
        col1, col2 = st.columns([3, 1])

        with col1:
            # Show relevance indicator in the ID line
            relevance_indicator = "📌 " if score > 0 else "⚪ "
            st.caption(f"{relevance_indicator}{doc_id}")
            st.write(doc_text)
            if explanation:
                st.caption(f"*{explanation}*")

        with col2:
            # Score display - differentiate between true documents and predictions
            if is_true_document:
                st.markdown(f":blue[Score: **{int(score)}**]")
            else:
                st.markdown(f"Similarity: **{score:.2f}**")

            # True relevance if available (only for predictions)
            if true_relevance is not None and not is_true_document:
                st.markdown(f"True relevance: **{true_relevance}**")

        # Add a subtle separator
        st.markdown("<hr style='margin: 5px 0; opacity: 0.2'>", unsafe_allow_html=True)


def display_metrics_summary(result: Dict[str, Any]) -> None:
    """Display evaluation metrics in a standardized format.

    Args:
        result: Result dictionary from evaluate_embeddings()
    """
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
