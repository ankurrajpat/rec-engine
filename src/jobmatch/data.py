"""Resume-to-job fit data: loading, cleaning, SQLite store.

Dataset: cnamuangtoun/resume-job-description-fit (Hugging Face). Each row is a
(resume, job description, label) triple with label in {No Fit, Potential Fit, Good Fit}.
test.csv holds 71 job descriptions that never appear in train.csv, so evaluating on it
simulates "a brand-new posting arrives and must be ranked against known candidates".
"""
import hashlib
import re
import sqlite3
from pathlib import Path

import pandas as pd

GRADE = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
BASE = "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/"


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip()


def doc_id(prefix: str, text: str) -> str:
    return prefix + hashlib.md5(text.encode()).hexdigest()[:10]


def load_split(path) -> pd.DataFrame:
    d = pd.read_csv(path).drop_duplicates(["resume_text", "job_description_text"])
    d["resume_text"] = d.resume_text.map(clean)
    d["job_text"] = d.pop("job_description_text").map(clean)
    d["rid"] = d.resume_text.map(lambda t: doc_id("r", t))
    d["jid"] = d.job_text.map(lambda t: doc_id("j", t))
    d["grade"] = d.label.map(GRADE)
    return d.reset_index(drop=True)


def to_sqlite(df: pd.DataFrame, db_path, split: str):
    """Store candidates, jobs and labelled pairs in normalised tables (3NF)."""
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS candidates(rid TEXT PRIMARY KEY, text TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(jid TEXT PRIMARY KEY, text TEXT NOT NULL, split TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS fit_labels(
            jid TEXT NOT NULL REFERENCES jobs(jid),
            rid TEXT NOT NULL REFERENCES candidates(rid),
            grade INTEGER NOT NULL CHECK(grade IN (0,1,2)),
            PRIMARY KEY(jid, rid));
        CREATE INDEX IF NOT EXISTS ix_labels_rid ON fit_labels(rid);
        """
    )
    con.executemany("INSERT OR IGNORE INTO candidates VALUES(?,?)",
                    df[["rid", "resume_text"]].drop_duplicates("rid").itertuples(index=False))
    con.executemany("INSERT OR IGNORE INTO jobs VALUES(?,?,?)",
                    [(r.jid, r.job_text, split) for r in df.drop_duplicates("jid").itertuples()])
    con.executemany("INSERT OR IGNORE INTO fit_labels VALUES(?,?,?)",
                    df[["jid", "rid", "grade"]].itertuples(index=False))
    con.commit()
    con.close()
