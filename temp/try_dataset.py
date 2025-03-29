import pytest
import json
from emb_eval_toybox.data.dataset import SearchDataset


@pytest.fixture
def sample_data(tmp_path):
    """Create a temporary sample dataset for testing."""
    data = {
        "queries": [
            "What's a strong espresso drink with milk?",
            "I want something sweet with coffee",
        ],
        "documents": [
            "A cappuccino is made with equal parts espresso, steamed milk, and milk foam.",
            "A latte is an espresso-based drink with steamed milk and a small layer of foam.",
            "A mocha combines espresso with steamed milk and chocolate, topped with whipped cream.",
        ],
        "relevance": [
            [1, 1, 0],  # First query: cappuccino and latte are relevant
            [0, 0, 1],  # Second query: mocha is relevant
        ],
    }

    # Create temporary JSON file
    data_path = tmp_path / "test_dataset.json"
    with open(data_path, "w") as f:
        json.dump(data, f)

    return data_path


def test_dataset_initialization(sample_data):
    """Test basic dataset initialization."""
    dataset = SearchDataset(sample_data)

    assert len(dataset.queries) == 2
    assert len(dataset.documents) == 3
    assert dataset.relevance.shape == (2, 3)


def test_get_relevant_documents(sample_data):
    """Test getting relevant documents for queries."""
    dataset = SearchDataset(sample_data)

    # Test first query (should have two relevant documents)
    relevant_docs = dataset.get_relevant_documents(0)
    assert len(relevant_docs) == 2
    assert 0 in relevant_docs  # cappuccino
    assert 1 in relevant_docs  # latte

    # Test second query (should have one relevant document)
    relevant_docs = dataset.get_relevant_documents(1)
    assert len(relevant_docs) == 1
    assert 2 in relevant_docs  # mocha


def test_dataset_length(sample_data):
    """Test dataset length."""
    dataset = SearchDataset(sample_data)
    assert len(dataset) == 2  # Number of queries


def test_dataset_indexing(sample_data):
    """Test dataset indexing."""
    dataset = SearchDataset(sample_data)

    # Test first query
    query, relevance = dataset[0]
    assert query == "What's a strong espresso drink with milk?"
    assert relevance.shape == (3,)  # Number of documents
    assert relevance[0] == 1  # cappuccino is relevant
    assert relevance[1] == 1  # latte is relevant
    assert relevance[2] == 0  # mocha is not relevant


def test_invalid_file_path():
    """Test handling of invalid file path."""
    with pytest.raises(FileNotFoundError):
        SearchDataset("nonexistent_file.json")


def test_invalid_data_format(tmp_path):
    """Test handling of invalid data format."""
    # Create a file with invalid JSON
    invalid_path = tmp_path / "invalid.json"
    with open(invalid_path, "w") as f:
        f.write("invalid json")

    with pytest.raises(json.JSONDecodeError):
        SearchDataset(invalid_path)


def test_missing_required_fields(tmp_path):
    """Test handling of missing required fields."""
    # Create a file with missing fields
    invalid_data = {
        "queries": ["test query"],
        # missing documents and relevance
    }

    invalid_path = tmp_path / "missing_fields.json"
    with open(invalid_path, "w") as f:
        json.dump(invalid_data, f)

    with pytest.raises(KeyError):
        SearchDataset(invalid_path)
