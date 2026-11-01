"""Ranking metrics and full-catalog evaluation loop."""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def recall_at_k(recommended, actual, k):
    """Fraction of the user's held-out items that appear in the top-k list."""
    if not actual:
        return np.nan
    return len(set(recommended[:k]) & set(actual)) / len(actual)


def ndcg_at_k(recommended, actual, k):
    """Normalized discounted cumulative gain at k (binary relevance)."""
    if not actual:
        return np.nan
    actual = set(actual)
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in actual)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / ideal


def _train_csr(train: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    return csr_matrix(
        (np.ones(len(train)), (train.user.values, train.item.values)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )


def evaluate(score_fn, train: pd.DataFrame, test: pd.DataFrame,
             n_users: int, n_items: int, ks=(10, 50), batch_size: int = 512):
    """Full-catalog ranking evaluation.

    score_fn(user_idx_batch) -> (batch, n_items) score matrix.
    Items the user already consumed in train are masked out before ranking.
    Metrics are macro-averaged over users that have at least one test event.
    """
    train_mat = _train_csr(train, n_users, n_items)
    test_by_user = defaultdict(list)
    for u, i in zip(test.user.values, test.item.values):
        test_by_user[u].append(i)
    users = sorted(test_by_user)
    k_max = max(ks)
    hits = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = score_fn(np.asarray(batch))
        for row, u in enumerate(batch):
            seen = train_mat.indices[train_mat.indptr[u]:train_mat.indptr[u + 1]]
            scores[row, seen] = -np.inf
        top = np.argpartition(-scores, k_max, axis=1)[:, :k_max]
        order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)
        for row, u in enumerate(batch):
            actual = test_by_user[u]
            recs = top[row].tolist()
            for k in ks:
                hits[k].append(recall_at_k(recs, actual, k))
                ndcgs[k].append(ndcg_at_k(recs, actual, k))
    out = {"n_users_eval": len(users), "n_test_events": int(len(test))}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(hits[k]))
        out[f"ndcg@{k}"] = float(np.mean(ndcgs[k]))
    return out
