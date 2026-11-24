"""Graded ranking metrics, computed per job (query) over its labelled candidate list."""
import numpy as np


def dcg(g, k):
    g = np.asarray(g, float)[:k]
    return float(((2 ** g - 1) / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg_at_k(grades_in_rank_order, k):
    ideal = dcg(sorted(grades_in_rank_order, reverse=True), k)
    return dcg(grades_in_rank_order, k) / ideal if ideal > 0 else np.nan


def precision_good_at_k(grades_in_rank_order, k):
    top = list(grades_in_rank_order)[:k]
    return sum(g == 2 for g in top) / max(len(top), 1)


def average_precision(grades_in_rank_order, min_grade=2):
    hits, s = 0, 0.0
    for i, g in enumerate(grades_in_rank_order, 1):
        if g >= min_grade:
            hits += 1
            s += hits / i
    return s / hits if hits else np.nan


def per_job_metrics(df, score_col, ks=(5, 10)):
    """df has columns jid, grade, <score_col>. Jobs with no relevant candidate are skipped (NaN)."""
    rows = {}
    for jid, g in df.groupby("jid"):
        order = g.sort_values(score_col, ascending=False, kind="stable").grade.tolist()
        r = {f"ndcg@{k}": ndcg_at_k(order, k) for k in ks}
        r.update({f"p_good@{k}": precision_good_at_k(order, k) for k in ks})
        r["map_good"] = average_precision(order)
        rows[jid] = r
    import pandas as pd
    return pd.DataFrame(rows).T


def summarize(m, n_boot=1000, seed=0):
    """Mean over jobs with 95% bootstrap CI (resampling jobs)."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in m.columns:
        v = m[c].dropna().to_numpy()
        boots = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
        out[c] = (v.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    return out
