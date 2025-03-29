import pytest
import numpy as np
from ..evaluation import (
    calculate_dcg,
    calculate_ndcg,
    calculate_precision_recall,
)

# Test data
@pytest.fixture
def sample_relevance_scores():
    return [3, 2, 1, 0, 2]

@pytest.fixture
def sample_ideal_scores():
    return [3, 2, 2, 1, 0]

def test_calculate_dcg():
    # Test with perfect ordering
    # For scores [3,2,1]:
    # (2^3 - 1) + (2^2 - 1)/log2(3) + (2^1 - 1)/log2(4)
    # = 7 + 3/1.58 + 1/2 = 7 + 1.89 + 0.5 ≈ 9.39
    assert calculate_dcg([3, 2, 1]) == pytest.approx(9.39, rel=1e-2)

    # Test simpler case
    # For scores [1, 1]:
    # (2^1 - 1) + (2^1 - 1)/log2(3)
    # = 1 + 1/1.58 ≈ 1.63
    assert calculate_dcg([1, 1]) == pytest.approx(1.63, rel=1e-2)

    # Test with k limit
    scores = [3, 2, 1, 0, 2]
    assert calculate_dcg(scores, k=3) == calculate_dcg(scores[:3])

    # Test with empty list
    assert calculate_dcg([]) == 0.0

def test_calculate_ndcg(sample_relevance_scores, sample_ideal_scores):
    # Test perfect ordering (should be 1.0)
    assert calculate_ndcg([3, 2, 1], [3, 2, 1]) == pytest.approx(1.0)

    # Test with different ordering
    ndcg = calculate_ndcg(sample_relevance_scores, sample_ideal_scores, k=3)
    assert 0 <= ndcg <= 1.0

    # Test with zero ideal scores
    assert calculate_ndcg([1, 0, 0], [0, 0, 0]) == 0.0

def test_calculate_precision_recall():
    # Test perfect prediction
    predicted = [0, 1, 2]
    true_relevant = [0, 1, 2]
    precision, recall = calculate_precision_recall(predicted, true_relevant)
    assert precision == 1.0
    assert recall == 1.0

    # Test partial match
    predicted = [0, 1, 3, 4]
    true_relevant = [0, 1, 2]
    precision, recall = calculate_precision_recall(predicted, true_relevant)
    assert precision == 0.5  # 2 correct out of 4 predicted
    assert recall == pytest.approx(2/3)  # 2 found out of 3 relevant

    # Test with k limit
    precision, recall = calculate_precision_recall(predicted, true_relevant, k=2)
    assert precision == 1.0  # First 2 predictions are correct
    assert recall == pytest.approx(2/3)  # 2 found out of 3 relevant

    # Test empty cases
    assert calculate_precision_recall([], []) == (0.0, 1.0)
    assert calculate_precision_recall([1, 2], []) == (0.0, 1.0)
    assert calculate_precision_recall([], [1, 2]) == (0.0, 0.0)
