import pandas as pd

from src import data


def _toy():
    return pd.DataFrame({
        "user": [0, 0, 1, 1, 2],
        "item": [0, 1, 1, 2, 0],
        "rating": [5, 4, 3, 4, 2],
        "timestamp": [10, 20, 30, 40, 50],
    })


def test_chronological_split_ordering():
    train, test, cutoff = data.chronological_split(_toy(), frac=0.6)
    assert train.timestamp.max() < cutoff <= test.timestamp.min()
    assert len(train) + len(test) == 5


def test_filter_seen_drops_cold_rows():
    train, test, _ = data.chronological_split(_toy(), frac=0.6)
    warm = data.filter_seen(train, test)
    assert set(warm.user) <= set(train.user)
    assert set(warm.item) <= set(train.item)
