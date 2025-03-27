import json
from pathlib import Path
from typing import List, Dict, Union
import numpy as np

class SearchDataset:
    def __init__(self, data_path: Union[str, Path]):
        """Initialize the dataset with the path to the JSON file.

        Args:
            data_path: Path to the JSON file containing the dataset.
                      Expected format:
                      {
                          "queries": List[str],
                          "documents": List[str],
                          "relevance": List[List[int]]
                      }
        """
        self.data_path = Path(data_path)
        self.queries: List[str] = []
        self.documents: List[str] = []
        self.relevance: np.ndarray = np.array([])

        self._load_data()

    def _load_data(self) -> None:
        """Load the dataset from JSON file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        self.queries = data['queries']
        self.documents = data['documents']
        self.relevance = np.array(data['relevance'])

    def get_relevant_documents(self, query_idx: int) -> List[int]:
        """Get indices of relevant documents for a given query."""
        return [i for i, rel in enumerate(self.relevance[query_idx]) if rel == 1]

    def __len__(self) -> int:
        """Return the number of queries in the dataset."""
        return len(self.queries)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:
        """Get a query and its relevance scores by index."""
        return self.queries[idx], self.relevance[idx]
