import json
from pathlib import Path
from typing import List, Dict, Union, Tuple
import numpy as np


class SearchDataset:
    """Dataset for evaluating search/retrieval performance.

    Supports two types of evaluation:
    - "basic": Binary relevance (0/1) for precision/recall metrics
    - "ndcg": Graded relevance (0-N) for normalized discounted cumulative gain
    """
    def __init__(self, data_path: Union[str, Path]):
        """Initialize the dataset with the path to the JSON file.

        Args:
            data_path: Path to the JSON file containing the dataset.
                      Expected format:
                      {
                          "metadata": {
                              "name": str,
                              "evaluation_type": str,  # "basic" (precision/recall) or "ndcg"
                              "description": str
                          },
                          "queries": List[str],
                          "documents": List[str],
                          "relevance": List[List[int]]
                      }
        """
        self.data_path = Path(data_path)
        self.queries: List[str] = []
        self.documents: List[str] = []
        self.relevance: np.ndarray = np.array([])
        self.metadata: Dict[str, str] = {}

        self._load_data()

    def _load_data(self) -> None:
        """Load the dataset from JSON file."""
        try:
            with open(self.data_path, "r") as f:
                data = json.load(f)

            # Set default metadata if not provided
            self.metadata = data.get("metadata", {})
            if "evaluation_type" not in self.metadata:
                self.metadata["evaluation_type"] = "basic"
            if "description" not in self.metadata:
                self.metadata["description"] = "No description provided"
            if "name" not in self.metadata:
                self.metadata["name"] = self.data_path.stem

            # Load data fields
            self.queries = data["queries"]
            self.documents = data["documents"]

            # Convert relevance scores to numpy array for efficiency
            self.relevance = np.array(data["relevance"])

            # Validate data consistency
            if len(self.queries) != len(self.relevance):
                raise ValueError(
                    f"Number of queries ({len(self.queries)}) doesn't match "
                    f"number of relevance score lists ({len(self.relevance)})"
                )

            if any(len(rel) != len(self.documents) for rel in self.relevance):
                raise ValueError(
                    "Each relevance score list must have the same length as documents list"
                )

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

    def __len__(self) -> int:
        """Return the number of queries in the dataset."""
        return len(self.queries)

    def __getitem__(self, idx: int) -> Tuple[str, List[int]]:
        """Get a query and indices of its relevant documents by index."""
        return self.queries[idx], self.get_relevant_documents(idx)
