"""Cold-start analysis: how does each model degrade for low-activity users
and rarely-rated items? Writes results/cold_start.json + a figure."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src import config
from src.models.two_tower import two_tower_score_fn

USER_BUCKETS = [(1, 20, "1-20"), (20, 50, "21-50"), (50, 100, "51-100"), (100, 10_000, "101+")]
ITEM_BUCKETS = [(1, 10, "1-10"), (10, 50, "11-50"), (50, 200, "51-200"), (200, 100_000, "201+")]


def bucket_of(v, buckets):
    for lo, hi, label in buckets:
        if lo <= v < hi:
            return label
    return None


def recall_by_bucket(score_fn, train, test, n_items, ks=(10,)):
    train_mat = csr_matrix((np.ones(len(train)), (train.user.values, train.item.values)),
                           shape=(int(train.user.max()) + 1, n_items), dtype=np.float32)
    test_by_user = defaultdict(list)
    for u, i in zip(test.user.values, test.item.values):
        test_by_user[u].append(i)
    users = sorted(test_by_user)
    k_max = max(ks)
    per_user_recs = {}
    for s in range(0, len(users), 512):
        batch = users[s:s + 512]
        scores = score_fn(np.asarray(batch))
        for row, u in enumerate(batch):
            seen = train_mat.indices[train_mat.indptr[u]:train_mat.indptr[u + 1]]
            scores[row, seen] = -np.inf
        top = np.argpartition(-scores, k_max, axis=1)[:, :k_max]
        order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)
        for row, u in enumerate(batch):
            per_user_recs[u] = top[row].tolist()
    return per_user_recs, test_by_user


def main():
    train = pd.read_csv(config.ARTIFACTS_DIR / "train.csv.gz")
    test_warm = pd.read_csv(config.ARTIFACTS_DIR / "test_warm.csv.gz")
    user_emb = np.load(config.ARTIFACTS_DIR / "user_embeddings.npy")
    item_emb = np.load(config.ARTIFACTS_DIR / "item_embeddings.npy")
    pop_scores = np.load(config.ARTIFACTS_DIR / "popularity_scores.npy")
    n_items = item_emb.shape[0]

    user_activity = train.user.value_counts()
    item_popularity = train.item.value_counts()

    models = {
        "popularity": lambda u: np.tile(pop_scores, (len(u), 1)).astype(np.float32),
        "two_tower": two_tower_score_fn(user_emb, item_emb),
    }
    out = {"by_user_activity": {}, "by_item_popularity": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for name, fn in models.items():
        recs, test_by_user = recall_by_bucket(fn, train, test_warm, n_items, ks=(10,))
        # user-activity buckets: average recall@10 over users in each bucket
        ub = defaultdict(list)
        for u, actual in test_by_user.items():
            b = bucket_of(user_activity.get(u, 0), USER_BUCKETS)
            if b:
                hits = len(set(recs[u][:10]) & set(actual)) / len(actual)
                ub[b].append(hits)
        out["by_user_activity"][name] = {b: round(float(np.mean(v)), 4) for b, v in ub.items()}
        # item-popularity buckets: recall@10 over test events, grouped by item
        ib = defaultdict(lambda: [0, 0])
        for u, actual in test_by_user.items():
            recset = set(recs[u][:10])
            for i in actual:
                b = bucket_of(item_popularity.get(i, 0), ITEM_BUCKETS)
                if b:
                    ib[b][1] += 1
                    ib[b][0] += int(i in recset)
        out["by_item_popularity"][name] = {
            b: round(hit / tot, 4) for b, (hit, tot) in ib.items() if tot > 0
        }

    labels_u = [b[2] for b in USER_BUCKETS]
    labels_i = [b[2] for b in ITEM_BUCKETS]
    width = 0.35
    for j, name in enumerate(models):
        axes[0].bar(np.arange(len(labels_u)) + j * width,
                    [out["by_user_activity"][name].get(l, 0) for l in labels_u],
                    width, label=name)
        axes[1].bar(np.arange(len(labels_i)) + j * width,
                    [out["by_item_popularity"][name].get(l, 0) for l in labels_i],
                    width, label=name)
    axes[0].set_xticks(np.arange(len(labels_u)) + width / 2, labels_u)
    axes[0].set_title("Recall@10 by user train activity")
    axes[0].set_xlabel("# train ratings")
    axes[0].legend()
    axes[1].set_xticks(np.arange(len(labels_i)) + width / 2, labels_i)
    axes[1].set_title("Recall@10 by item train popularity")
    axes[1].set_xlabel("# train ratings of the held-out item")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cold_start.png", dpi=110)

    (config.RESULTS_DIR / "cold_start.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
