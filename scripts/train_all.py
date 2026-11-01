"""Staged training pipeline: prepare -> popularity -> mf -> two_tower -> finalize.

Each stage persists its outputs so long runs can be split across invocations.
This is the single entry point that produces every number in the README.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src import config, data, tracking
from src.eval import evaluate
from src.models.mf import mf_score_fn, train_mf
from src.models.popularity import PopularityModel
from src.models.two_tower import (
    export_item_embeddings,
    export_user_embeddings,
    genre_multihot,
    train_two_tower,
    two_tower_score_fn,
)

MF_PARAMS = dict(dim=config.EMB_DIM, epochs=10, negatives=4, batch_size=8192,
                 lr=0.02, max_per_epoch=400_000, seed=config.SEED)
TT_PARAMS = dict(dim=config.EMB_DIM, epochs=16, batch_size=2048, lr=3e-3,
                 max_per_epoch=400_000, seed=config.SEED)

PARTIAL = config.ARTIFACTS_DIR / "metrics_partial.json"


def _load_partial():
    if PARTIAL.exists():
        return json.loads(PARTIAL.read_text())
    return {}


def _save_partial(d):
    PARTIAL.write_text(json.dumps(d, indent=2))


def _meta():
    ratings = data.load_ratings()
    train = pd.read_csv(config.ARTIFACTS_DIR / "train.csv.gz")
    test_warm = pd.read_csv(config.ARTIFACTS_DIR / "test_warm.csv.gz")
    n_users = int(ratings.user.max()) + 1
    n_items = int(ratings.item.max()) + 1
    return train, test_warm, n_users, n_items


def stage_prepare():
    ratings = data.load_ratings()
    train, test, cutoff = data.chronological_split(ratings)
    test_warm = data.filter_seen(train, test)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    train.to_csv(config.ARTIFACTS_DIR / "train.csv.gz", index=False, compression="gzip")
    test.to_csv(config.ARTIFACTS_DIR / "test.csv.gz", index=False, compression="gzip")
    test_warm.to_csv(config.ARTIFACTS_DIR / "test_warm.csv.gz", index=False, compression="gzip")
    _save_partial({"cutoff_unix": cutoff, "train_events": len(train),
                   "test_events": len(test), "test_warm_events": len(test_warm)})
    print(f"train={len(train)} test={len(test)} test_warm={len(test_warm)}")


def stage_popularity():
    tracking.setup()
    train, test_warm, n_users, n_items = _meta()
    pop = PopularityModel().fit(train, n_items)
    m = evaluate(pop.score, train, test_warm, n_users, n_items, ks=config.TOP_KS)
    tracking.log_run("popularity", {"smoothing_m": 50}, m)
    np.save(config.ARTIFACTS_DIR / "popularity_scores.npy", pop.scores.astype(np.float32))
    p = _load_partial()
    p.setdefault("metrics", {})["popularity"] = m
    _save_partial(p)
    print("popularity:", m)


def stage_mf():
    tracking.setup()
    train, test_warm, n_users, n_items = _meta()
    mf, hist = train_mf(train, n_users, n_items, **MF_PARAMS)
    torch.save(mf.state_dict(), config.ARTIFACTS_DIR / "mf_state_dict.pt")
    m = evaluate(mf_score_fn(mf), train, test_warm, n_users, n_items, ks=config.TOP_KS)
    tracking.log_run("matrix_factorization", MF_PARAMS, m, extra={"final_bce": hist["final_bce"]})
    p = _load_partial()
    p.setdefault("metrics", {})["matrix_factorization"] = m
    _save_partial(p)
    print("mf:", m)


def stage_two_tower():
    tracking.setup()
    train, test_warm, n_users, n_items = _meta()
    movies = data.load_movies()
    gf = genre_multihot(movies, n_items)
    tt, hist = train_two_tower(train, gf, n_users, n_items, **TT_PARAMS)
    user_emb = export_user_embeddings(tt, n_users)
    item_emb = export_item_embeddings(tt, gf, n_items)
    torch.save(tt.state_dict(), config.ARTIFACTS_DIR / "two_tower_state_dict.pt")
    np.save(config.ARTIFACTS_DIR / "user_embeddings.npy", user_emb)
    np.save(config.ARTIFACTS_DIR / "item_embeddings.npy", item_emb)
    m = evaluate(two_tower_score_fn(user_emb, item_emb), train, test_warm, n_users, n_items, ks=config.TOP_KS)
    tracking.log_run("two_tower", TT_PARAMS, m, extra={"final_ce": hist["final_ce"]})
    p = _load_partial()
    p.setdefault("metrics", {})["two_tower"] = m
    _save_partial(p)
    print("two_tower:", m)


def stage_finalize():
    p = _load_partial()
    ratings = data.load_ratings()
    p["n_users"] = int(ratings.user.max()) + 1
    p["n_items"] = int(ratings.item.max()) + 1
    p["mf_params"] = MF_PARAMS
    p["two_tower_params"] = TT_PARAMS
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(p, indent=2))
    print(json.dumps(p["metrics"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "popularity", "mf", "two_tower", "finalize"])
    args = ap.parse_args()
    t0 = time.time()
    {"prepare": stage_prepare, "popularity": stage_popularity, "mf": stage_mf,
     "two_tower": stage_two_tower, "finalize": stage_finalize}[args.stage]()
    print(f"[{args.stage}] {time.time() - t0:.1f}s", flush=True)
