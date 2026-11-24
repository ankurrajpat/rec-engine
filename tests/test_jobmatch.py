import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.jobmatch.data import load_split, to_sqlite
from src.jobmatch.evaluate import average_precision, ndcg_at_k, per_job_metrics, precision_good_at_k


def test_ndcg_perfect_and_worst():
    assert ndcg_at_k([2, 1, 0], 3) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 2], 3) < 1.0
    assert np.isnan(ndcg_at_k([0, 0, 0], 3))  # no relevant item: undefined, skipped in aggregates


def test_ndcg_hand_value():
    # gains 3,1 at ranks 1,2 vs ideal 3,1 -> 1 ; swapped -> (1 + 3/log2(3)) / (3 + 1/log2(3))
    swapped = ndcg_at_k([1, 2], 2)
    assert swapped == pytest.approx((1 + 3 / np.log2(3)) / (3 + 1 / np.log2(3)))


def test_precision_and_map():
    assert precision_good_at_k([2, 0, 2, 1], 4) == 0.5
    assert average_precision([2, 0, 2]) == pytest.approx((1 + 2 / 3) / 2)


def test_per_job_metrics_uses_score_order():
    df = pd.DataFrame({"jid": ["a"] * 3, "grade": [0, 1, 2], "s": [0.1, 0.5, 0.9]})
    assert per_job_metrics(df, "s").loc["a", "ndcg@5"] == pytest.approx(1.0)


def test_sqlite_schema_and_integrity(tmp_path):
    csv = tmp_path / "x.csv"
    pd.DataFrame({"resume_text": ["python  dev", "java dev"], "job_description_text": ["ml job", "ml job"],
                  "label": ["Good Fit", "No Fit"]}).to_csv(csv, index=False)
    df = load_split(csv)
    db = tmp_path / "t.db"
    to_sqlite(df, db, "train")
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    assert sorted(r[0] for r in con.execute("SELECT grade FROM fit_labels")) == [0, 2]
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO fit_labels VALUES('j','r',7)")


def test_matcher_end_to_end_if_artifacts_exist():
    from pathlib import Path
    art = Path(__file__).resolve().parents[1] / "artifacts" / "jobmatch"
    if not (art / "jobmatch.db").exists():
        pytest.skip("run scripts/run_jobmatch.py first")
    from src.jobmatch.matcher import JobMatcher
    m = JobMatcher(art, embed_fn=lambda t: np.stack([m.index.reconstruct(0)] * len(t)))
    out = m.match("python machine learning engineer " * 10, k_retrieve=20, k_return=5)
    assert len(out) == 5 and out.rerank_score.is_monotonic_decreasing
