import json
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any, Optional
import numpy as np


class SearchDataset:
    """Dataset for evaluating search/retrieval performance with document-centric structure.

    Supports two types of evaluation:
    - "basic": Binary relevance (0/1) for precision/recall metrics
    - "ndcg": Graded relevance (0-N) for normalized discounted cumulative gain
    """
    def __init__(self, data_path: Union[str, Path]):
        """Initialize the dataset with the path to the JSON file.

        Args:
            data_path: Path to the JSON file containing the dataset in document-centric format.
        """
        self.data_path = Path(data_path)
        self.queries: List[str] = []
        self.query_ids: List[str] = []
        self.documents: List[str] = []
        self.document_ids: List[str] = []
        self.relevance: np.ndarray = np.array([])
        self.explanations: Dict[str, Dict[str, str]] = {}
        self.metadata: Dict[str, Any] = {}

        self._load_data()

    def _load_data(self) -> None:
        """Load the dataset from JSON file in document-centric format."""
        try:
            with open(self.data_path, "r") as f:
                data = json.load(f)

            # Load metadata
            self.metadata = data.get("metadata", {})
            if "evaluation_type" not in self.metadata:
                self.metadata["evaluation_type"] = "basic"
            if "description" not in self.metadata:
                self.metadata["description"] = "No description provided"
            if "name" not in self.metadata:
                self.metadata["name"] = self.data_path.stem

            # Extract query data
            self.query_ids = [q["id"] for q in data["queries"]]
            self.queries = [q["text"] for q in data["queries"]]

            # Extract document data
            self.document_ids = [doc["id"] for doc in data["documents"]]
            self.documents = [doc["text"] for doc in data["documents"]]

            # Create relevance matrix and explanations dict
            self.relevance = np.zeros((len(self.queries), len(self.documents)), dtype=int)
            self.explanations = {qid: {} for qid in self.query_ids}

            # Fill in relevance matrix and explanations from document-centric structure
            for doc_idx, doc in enumerate(data["documents"]):
                for q_idx, q_id in enumerate(self.query_ids):
                    if q_id in doc.get("relevance", {}):
                        rel_data = doc["relevance"][q_id]
                        self.relevance[q_idx, doc_idx] = rel_data["score"]
                        if "explanation" in rel_data:
                            doc_id = doc["id"]
                            self.explanations[q_id][doc_id] = rel_data["explanation"]

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid dataset format in {self.data_path}: {e}")

    @property
    def evaluation_type(self) -> str:
        """Get the evaluation type this dataset is suited for."""
        return self.metadata["evaluation_type"]

    @property
    def description(self) -> str:
        """Get the dataset description."""
        return self.metadata["description"]

    @property
    def name(self) -> str:
        """Get the dataset name."""
        return self.metadata["name"]

    def get_relevant_documents(self, query_idx: int) -> List[int]:
        """Get indices of relevant documents for a given query."""
        return [i for i, rel in enumerate(self.relevance[query_idx]) if rel > 0]

    def get_explanation(self, query_idx: int, doc_idx: int) -> Optional[str]:
        """Get explanation for a relevance score if available."""
        query_id = self.query_ids[query_idx]
        doc_id = self.document_ids[doc_idx]

        return self.explanations.get(query_id, {}).get(doc_id)

    def get_query_id(self, query_idx: int) -> str:
        """Get query ID by index."""
        return self.query_ids[query_idx]

    def get_document_id(self, doc_idx: int) -> str:
        """Get document ID by index."""
        return self.document_ids[doc_idx]

    def __len__(self) -> int:
        """Return the number of queries in the dataset."""
        return len(self.queries)

    def __getitem__(self, idx: int) -> Tuple[str, List[int]]:
        """Get a query and indices of its relevant documents by index."""
        return self.queries[idx], self.get_relevant_documents(idx)
