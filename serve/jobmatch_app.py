"""FastAPI: POST /match {job_text, k} -> top candidates via dense retrieval + LambdaRank rerank.

Needs artifacts from `python scripts/run_jobmatch.py`. The embedding model loads on first use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ART = Path(__file__).resolve().parent.parent / "artifacts" / "jobmatch"
app = FastAPI(title="rec-engine job matching", version="0.1.0")
_matcher = None


class MatchRequest(BaseModel):
    job_text: str = Field(min_length=30)
    k: int = Field(10, ge=1, le=50)


def get_matcher():
    global _matcher
    if _matcher is None:
        from src.jobmatch.matcher import JobMatcher

        if not (ART / "reranker.txt").exists():
            raise HTTPException(503, "artifacts missing: run scripts/run_jobmatch.py")
        _matcher = JobMatcher(ART)
    return _matcher


@app.get("/health")
def health():
    return {"status": "ok", "artifacts": (ART / "reranker.txt").exists()}


@app.post("/match")
def match(req: MatchRequest):
    m = get_matcher()
    out = m.match(req.job_text, k_retrieve=50, k_return=req.k)
    return {"matches": out.to_dict("records")}
