import numpy as np

from src.eval import ndcg_at_k, recall_at_k


def test_recall_perfect():
    assert recall_at_k([1, 2, 3], [1, 2], 3) == 1.0


def test_recall_partial():
    assert recall_at_k([1, 9, 8], [1, 2], 3) == 0.5


def test_recall_zero():
    assert recall_at_k([4, 5], [1, 2], 2) == 0.0


def test_ndcg_perfect():
    # both relevant items at the top -> NDCG 1
    assert abs(ndcg_at_k([1, 2, 3], [1, 2], 3) - 1.0) < 1e-9


def test_ndcg_second_position():
    # one relevant item at position 2 of 1 relevant item
    expected = (1 / np.log2(3)) / (1 / np.log2(2))
    assert abs(ndcg_at_k([9, 1, 8], [1], 3) - expected) < 1e-9


def test_empty_actual_is_nan():
    assert np.isnan(recall_at_k([1], [], 5))
