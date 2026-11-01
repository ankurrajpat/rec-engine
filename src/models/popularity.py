"""Popularity baseline: rank items by train-set interaction count."""
import numpy as np
import pandas as pd


class PopularityModel:
    def __init__(self):
        self.scores = None

    def fit(self, train: pd.DataFrame, n_items: int) -> "PopularityModel":
        counts = np.bincount(train.item.values, minlength=n_items).astype(np.float64)
        # Bayesian smoothing toward the global mean shrinks items with a
        # handful of 5-star ratings below genuinely popular ones.
        sums = np.bincount(train.item.values, weights=train.rating.values, minlength=n_items)
        prior = train.rating.mean()
        m = 50.0  # smoothing strength
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = (sums + m * prior) / (counts + m)
        self.scores = np.nan_to_num(smoothed) * np.log1p(counts)
        return self

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        """Same ranking for every user; tiled across the batch."""
        return np.tile(self.scores, (len(user_idx), 1)).astype(np.float32)
