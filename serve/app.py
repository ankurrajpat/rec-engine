"""FastAPI serving layer: /recommend backed by the FAISS item index.

Endpoints:
  GET /health                       -> {"status": "ok", "items": N}
  GET /recommend?user_id=U&k=K      -> top-K for a known (warm) user
  GET /recommend?user_id=U&k=K with an unknown/cold user falls back to
  popularity and says so in the response ("fallback": "popularity").
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query

from src import config

app = FastAPI(title="rec-engine", version="0.1.0")

_user_emb = None
_item_emb = None
_pop = None
_index = None
_movies = None
_train = None
_id_maps = None


def _load():
    global _user_emb, _item_emb, _pop, _index, _movies, _train, _id_maps
    if _index is not None:
        return
    _user_emb = np.load(config.ARTIFACTS_DIR / "user_embeddings.npy")
    _item_emb = np.load(config.ARTIFACTS_DIR / "item_embeddings.npy").astype(np.float32)
    _pop = np.load(config.ARTIFACTS_DIR / "popularity_scores.npy")
    _index = faiss.read_index(str(config.ARTIFACTS_DIR / "item_index.faiss"))
    _movies = pd.read_csv(config.DATA_DIR / "movies.dat", sep="::", engine="python",
                          encoding="latin-1", names=["item_raw", "title", "genres"])
    _id_maps = json.loads((config.DATA_DIR / "id_maps.json").read_text())
    _train = pd.read_csv(config.ARTIFACTS_DIR / "train.csv.gz")


@app.on_event("startup")
def _startup():
    _load()


@app.get("/health")
def health():
    _load()
    return {"status": "ok", "items": int(_index.ntotal)}


def _titles(item_idx: np.ndarray, scores: np.ndarray):
    idx_to_raw = {int(v): int(k) for k, v in _id_maps["item_raw_to_idx"].items()}
    by_raw = _movies.set_index("item_raw")
    out = []
    for i, s in zip(item_idx, scores):
        raw = idx_to_raw.get(int(i))
        row = by_raw.loc[raw] if raw in by_raw.index else None
        out.append({
            "movie_id": raw,
            "title": None if row is None else row.title,
            "genres": None if row is None else row.genres,
            "score": round(float(s), 4),
        })
    return out


@app.get("/recommend")
def recommend(user_id: int = Query(..., description="Raw MovieLens user id"),
              k: int = Query(10, ge=1, le=100)):
    _load()
    user_map = _id_maps["user_raw_to_idx"]
    if str(user_id) in user_map:
        uidx = int(user_map[str(user_id)])
        vec = _user_emb[uidx:uidx + 1].astype(np.float32)
        scores, items = _index.search(vec, k + 200)
        # mask items already consumed in train
        seen = set(_train.loc[_train.user == uidx, "item"].values)
        keep = [(i, s) for i, s in zip(items[0], scores[0]) if i not in seen][:k]
        items_k = np.array([i for i, _ in keep])
        scores_k = np.array([s for _, s in keep])
        return {"user_id": user_id, "k": k, "fallback": None,
                "recommendations": _titles(items_k, scores_k)}
    top = np.argsort(-_pop)[:k]
    return {"user_id": user_id, "k": k, "fallback": "popularity",
            "recommendations": _titles(top, _pop[top])}
