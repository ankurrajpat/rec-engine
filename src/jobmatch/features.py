"""Scoring signals for (job, candidate) pairs: lexical, dense, and overlap features."""
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

TOK = re.compile(r"[a-z][a-z0-9+#.\-]{1,}")


def tok(t):
    return TOK.findall(t.lower())


def embed(texts, model_name="BAAI/bge-small-en-v1.5", max_len=512, bs=16):
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name, device="cpu")
    m.max_seq_length = max_len
    return m.encode(list(texts), batch_size=bs, normalize_embeddings=True, show_progress_bar=False)


class PairFeaturizer:
    """Fit lexical statistics on the candidate pool only (no label information)."""

    def __init__(self, resumes: dict, jobs: dict, resume_emb: dict, job_emb: dict):
        self.resumes, self.jobs, self.re, self.je = resumes, jobs, resume_emb, job_emb
        self.rids = list(resumes)
        self.tf = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, ngram_range=(1, 2),
                                  max_features=60000)
        self.tf.fit(list(resumes.values()) + list(jobs.values()))
        self.rt = {k: v for k, v in zip(self.rids, self.tf.transform([resumes[r] for r in self.rids]))}
        self.jt = {j: self.tf.transform([t])[0] for j, t in jobs.items()}
        self.rtok = {r: set(tok(t)) for r, t in resumes.items()}
        self.bm = BM25Okapi([tok(resumes[r]) for r in self.rids])
        self.bm_idx = {r: i for i, r in enumerate(self.rids)}
        self._bm_cache = {}

    def bm25(self, jid, rid):
        if jid not in self._bm_cache:
            q = tok(self.jobs[jid])
            self._bm_cache[jid] = self.bm.get_scores(q)
        s = self._bm_cache[jid]
        return (s[self.bm_idx[rid]] - s.mean()) / (s.std() + 1e-9)

    def signals(self, jid, rid):
        tfidf = float(self.jt[jid].multiply(self.rt[rid]).sum())
        dense = float(self.je[jid] @ self.re[rid])
        jt = set(tok(self.jobs[jid]))
        overlap = len(jt & self.rtok[rid]) / (len(jt) + 1)
        return dict(bm25=self.bm25(jid, rid), tfidf=tfidf, dense=dense, overlap=overlap)

    def features(self, jid, rid):
        s = self.signals(jid, rid)
        s["resume_len"] = np.log1p(len(self.resumes[rid]))
        s["job_len"] = np.log1p(len(self.jobs[jid]))
        return s
