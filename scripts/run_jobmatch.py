"""Resume-to-job ranking experiment. Usage: python scripts/run_jobmatch.py [--data DIR]

Downloads the dataset, embeds texts, scores baselines and a LightGBM LambdaRank reranker,
writes results/jobmatch_results.json and artifacts/jobmatch/*.
"""
import argparse
import json
import pickle
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.jobmatch.data import BASE, load_split, to_sqlite
from src.jobmatch.evaluate import per_job_metrics, summarize
from src.jobmatch.features import PairFeaturizer, embed

import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "jobmatch"))
ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
args = ap.parse_args()
D = Path(args.data)
D.mkdir(parents=True, exist_ok=True)
for f in ("train", "test"):
    if not (D / f"{f}.csv").exists():
        urllib.request.urlretrieve(BASE + f"{f}.csv", D / f"{f}.csv")

tr, te = load_split(D / "train.csv"), load_split(D / "test.csv")
assert not set(tr.jid) & set(te.jid), "train/test jobs must be disjoint"
resumes = {**dict(zip(tr.rid, tr.resume_text)), **dict(zip(te.rid, te.resume_text))}
jobs = {**dict(zip(tr.jid, tr.job_text)), **dict(zip(te.jid, te.job_text))}
print(f"train: {len(tr)} pairs, {tr.jid.nunique()} jobs | test: {len(te)} pairs, {te.jid.nunique()} jobs "
      f"| unique resumes {len(resumes)}")

cache = D / "emb.pkl"
if cache.exists():
    E = pickle.load(open(cache, "rb"))
else:
    t0 = time.time()
    ids = list(resumes) + list(jobs)
    vec = embed([resumes[i] if i in resumes else jobs[i] for i in ids], args.model)
    E = dict(zip(ids, vec))
    pickle.dump(E, open(cache, "wb"))
    print(f"embedded {len(ids)} texts in {time.time()-t0:.0f}s")

pf = PairFeaturizer(resumes, jobs, {r: E[r] for r in resumes}, {j: E[j] for j in jobs})


def featurize(df):
    X = pd.DataFrame([pf.features(j, r) for j, r in zip(df.jid, df.rid)])
    # within-job standardisation: a ranker only needs relative position inside one job's list
    for c in ("bm25", "tfidf", "dense", "overlap"):
        g = X[c].groupby(df.jid.values)
        X[c + "_z"] = ((X[c] - g.transform("mean")) / (g.transform("std") + 1e-9)).fillna(0)
    return X


t0 = time.time()
Xtr, Xte = featurize(tr), featurize(te)
print(f"features in {time.time()-t0:.0f}s", list(Xtr.columns))
rng = np.random.default_rng(0)
FEATS = list(Xtr.columns)

def with_scores(df, X, extra=None):
    d = df[["jid", "rid", "grade"]].copy()
    for c in ("bm25", "tfidf", "dense", "overlap"):
        d[c] = X[c].values
    d["random"] = rng.random(len(d))
    # reciprocal rank fusion of the three text signals (no learning)
    for c in ("bm25", "tfidf", "dense"):
        d["rk_" + c] = d.groupby("jid")[c].rank(ascending=False)
    d["rrf"] = sum(1 / (60 + d["rk_" + c]) for c in ("bm25", "tfidf", "dense"))
    return d

def fit(Xa, da, params):
    da = da.reset_index(drop=True)
    order = da.jid.argsort(kind="stable")
    Xs, ys, gs = Xa.iloc[order], da.grade.iloc[order], da.jid.iloc[order]
    sizes = gs.groupby(gs.values, sort=False).size().values
    m = lgb.LGBMRanker(objective="lambdarank", n_jobs=1, verbose=-1, random_state=0, **params)
    m.fit(Xs, ys, group=sizes)
    return m

grid = [dict(n_estimators=n, learning_rate=lr, num_leaves=l, min_child_samples=20, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8)
        for n in (100, 300) for lr in (0.03, 0.1) for l in (4, 8)]
# model selection: 5-fold CV over *jobs* on train only (test never touched)
gkf = GroupKFold(5)
cv = []
for p in grid:
    sc = []
    for a, b in gkf.split(Xtr, tr.grade, tr.jid):
        m = fit(Xtr.iloc[a], tr.iloc[a], p)
        d = tr.iloc[b][["jid", "grade"]].copy()
        d["s"] = m.predict(Xtr.iloc[b])
        sc.append(per_job_metrics(d, "s")["ndcg@10"].mean())
    cv.append(float(np.mean(sc)))
best = grid[int(np.argmax(cv))]
print("CV ndcg@10 per config:", [round(c, 4) for c in cv], "-> best", best)

model = fit(Xtr, tr, best)
dte = with_scores(te, Xte)
dte["reranker"] = model.predict(Xte)
dtr_cvpred = None

systems = ["random", "tfidf", "bm25", "dense", "rrf", "reranker"]
M = {s: per_job_metrics(dte, s) for s in systems}
res = {"n_test_jobs": int(te.jid.nunique()), "n_test_pairs": int(len(te)),
       "n_train_jobs": int(tr.jid.nunique()), "n_train_pairs": int(len(tr)),
       "label_share_test": te.label.value_counts(normalize=True).round(3).to_dict(),
       "best_params": best, "cv_ndcg10": cv, "embedding_model": args.model, "systems": {}}
for s in systems:
    res["systems"][s] = {k: [round(x, 4) for x in v] for k, v in summarize(M[s]).items()}

# paired bootstrap of reranker minus each baseline on NDCG@10 over jobs
def paired(a, b, col="ndcg@10", n=2000):
    d = (M[a][col] - M[b][col]).dropna().to_numpy()
    bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return [round(d.mean(), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
res["paired_ndcg10_reranker_minus"] = {b: paired("reranker", b) for b in ("dense", "bm25", "tfidf", "rrf")}
imp = dict(zip(FEATS, model.booster_.feature_importance("gain")))
res["feature_gain"] = {k: round(float(v / sum(imp.values())), 3) for k, v in sorted(imp.items(), key=lambda x: -x[1])}

(ROOT / "results").mkdir(exist_ok=True)
json.dump(res, open(ROOT / "results" / "jobmatch_results.json", "w"), indent=2)
print(json.dumps(res, indent=1))

A = ROOT / "artifacts" / "jobmatch"
A.mkdir(parents=True, exist_ok=True)
model.booster_.save_model(str(A / "reranker.txt"))
db = A / "jobmatch.db"
if db.exists():
    db.unlink()
to_sqlite(tr, db, "train")
to_sqlite(te, db, "test")
ids = list(resumes) + list(jobs)
np.savez_compressed(A / "embeddings.npz", ids=np.array(ids), vec=np.stack([E[i] for i in ids]).astype("float16"))
