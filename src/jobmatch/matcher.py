"""Two-stage matcher: dense retrieval (FAISS) over the candidate pool, then LightGBM LambdaRank rerank."""
import sqlite3
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import PairFeaturizer, embed

Z_COLS = ("bm25", "tfidf", "dense", "overlap")


def add_within_list_z(X: pd.DataFrame) -> pd.DataFrame:
    """Standardise signals within one job's candidate list (same transform used in training)."""
    for c in Z_COLS:
        X[c + "_z"] = ((X[c] - X[c].mean()) / (X[c].std(ddof=1) + 1e-9)).fillna(0) if len(X) > 1 else 0.0
    return X


class JobMatcher:
    def __init__(self, art_dir, embed_fn=embed, feature_order=None):
        art = Path(art_dir)
        con = sqlite3.connect(art / "jobmatch.db")
        cand = dict(con.execute("SELECT rid, text FROM candidates"))
        jobs = dict(con.execute("SELECT jid, text FROM jobs"))
        con.close()
        z = np.load(art / "embeddings.npz", allow_pickle=True)
        self.ids = [str(i) for i in z["ids"]]
        vec = z["vec"].astype("float32")
        emb = dict(zip(self.ids, vec))
        self.cand_ids = list(cand)
        self.index = faiss.IndexFlatIP(vec.shape[1])
        self.index.add(np.stack([emb[r] for r in self.cand_ids]))
        self.pf = PairFeaturizer(cand, jobs, {r: emb[r] for r in cand}, {j: emb[j] for j in jobs})
        self.embed_fn = embed_fn
        self.ranker = lgb.Booster(model_file=str(art / "reranker.txt"))
        self.feature_order = self.ranker.feature_name()

    def match(self, job_text: str, k_retrieve: int = 50, k_return: int = 10):
        jid = "new_job"
        qv = self.embed_fn([job_text])[0].astype("float32")
        self.pf.jobs[jid] = job_text
        self.pf.je[jid] = qv
        self.pf.jt[jid] = self.pf.tf.transform([job_text])[0]
        self.pf._bm_cache.pop(jid, None)
        sims, idx = self.index.search(qv[None, :], min(k_retrieve, len(self.cand_ids)))
        rids = [self.cand_ids[i] for i in idx[0]]
        X = add_within_list_z(pd.DataFrame([self.pf.features(jid, r) for r in rids]))
        score = self.ranker.predict(X[self.feature_order])
        out = pd.DataFrame({"rid": rids, "dense_sim": sims[0], "rerank_score": score})
        return out.sort_values("rerank_score", ascending=False).head(k_return).reset_index(drop=True)
