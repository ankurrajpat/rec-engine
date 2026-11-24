# rec-engine

An end-to-end movie recommendation system built from scratch on **MovieLens 1M**:
popularity baseline -> matrix factorization -> two-tower neural retrieval, with a
chronological evaluation harness, cold-start analysis, FAISS serving and a Streamlit demo.

![Streamlit demo](docs/streamlit_demo.png)

## Problem

Given a user's rating history, rank the ~3.7k-movie catalog so the movies they will
rate next land near the top. Evaluated the way a production system is used: **train on
the past, predict the future** (global chronological split, not a random shuffle).

## Dataset

MovieLens 1M: 1,000,209 ratings from 6,040 users on 3,706 movies (density 4.5%).
Split at the 80th percentile of the timeline: 800,164 train events / 200,045 test events.
95,593 test events (48%) involve a user or item with no train history at all - they are
excluded from the main metrics and studied separately in the cold-start analysis.

![EDA](docs/eda.png)

## Results (full-catalog ranking, 1,143 evaluated users, 104,452 held-out events)

| Model                  | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
|------------------------|-----------|---------|-----------|---------|
| Popularity (smoothed)  | **0.0411** | **0.2650** | **0.1382** | **0.2289** |
| Matrix factorization   | 0.0333    | 0.1845  | 0.1281    | 0.1882  |
| Two-tower (in-batch)   | 0.0086    | 0.0534  | 0.0488    | 0.0667  |

All numbers produced by `scripts/train_all.py` on this repo's exact code; runs are
logged to MLflow (sqlite backend) and `results/runs.jsonl`.

**How to read these numbers.** Each evaluated user has ~91 held-out events on average,
so Recall@10 denominators are large (NDCG@10 exceeding Recall@10 is expected here, not a
bug). The chronological tail of MovieLens is dominated by blockbusters, which makes the
popularity baseline genuinely hard to beat on this split: MF comes close on Recall@50 but
does not pass it, and the lightly-trained two-tower lags on the head metric.

**Where the two-tower wins: the long tail.** Popularity gets *every one* of its hits on
items with 201+ train ratings and literally 0.0 recall on anything less popular. The
two-tower retrieves items with as few as 1-10 train ratings (Recall@10 0.0192 in that
bucket), which is exactly what a retrieval model is for.

![Cold start](docs/cold_start.png)

## Architecture

```
ratings.dat ──> chronological split ──> popularity baseline
                                     ├─> MF (biased, BCE + sampled negatives)
                                     └─> two-tower (user id-emb; item id-emb + genre projection;
                                                    in-batch sampled softmax, learned temperature)
                                                   │
        evaluation (Recall@K / NDCG@K, train-masked, full catalog)
                                                   │
        item embeddings ──> FAISS IndexFlatIP ──> FastAPI /recommend ──> Streamlit demo
```

- **MF**: user/item embeddings + biases, BCE over uniformly sampled negatives
  (dim 64, 10 epochs, 4 negatives/positive, lr 0.02).
- **Two-tower**: normalized 64-d towers; the item tower adds a linear projection of the
  multi-hot genre vector, so a brand-new item with known genres gets a sensible vector.
  Trained with in-batch sampled softmax + learnable temperature (16 epochs, batch 2048).
- **Serving**: user vector -> FAISS inner-product search -> train-history masking ->
  titles. Unknown users fall back to popularity (response says `"fallback": "popularity"`).
- **Tracking**: every run logs params + metrics to MLflow (`mlflow.db`) and
  `results/runs.jsonl`.

## Job matching track (resume-to-job ranking)

A second pipeline in `src/jobmatch/` applies the same retrieve-then-rank design to hiring:
given a new job description, rank candidate resumes.

**Data.** [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
(Hugging Face; no license stated on the dataset card, so the data is downloaded by the script and not
redistributed here). 8,000 labelled (resume, job) pairs, 643 resumes, 351 jobs, labels No Fit / Potential Fit /
Good Fit (graded 0/1/2). The official `test.csv` contains **71 jobs that never appear in train**, so the test
simulates a brand-new posting arriving. Hyper-parameters were chosen by 5-fold CV grouped by job on train only.

**Pipeline.** BGE-small embeddings (`BAAI/bge-small-en-v1.5`) in a FAISS index -> top-50 retrieval -> LightGBM
LambdaRank reranker over BM25, TF-IDF, dense cosine, keyword overlap and length features (also standardised within each
job's list). Pairs and labels live in normalised SQLite tables (`candidates`, `jobs`, `fit_labels`; see
`scripts/jobmatch_sql.sql`). Served by `serve/jobmatch_app.py` (`POST /match`).

**Results** (71 held-out jobs, 1,759 labelled pairs, mean over jobs, 95% bootstrap CI over jobs):

| System | NDCG@5 | NDCG@10 | MAP (Good Fit) |
|---|---|---|---|
| Random | 0.527 | 0.578 (0.524-0.629) | 0.571 |
| BM25 | 0.598 | 0.628 (0.568-0.687) | 0.605 |
| TF-IDF | 0.620 | 0.654 (0.591-0.715) | 0.662 |
| Dense (BGE-small) | 0.703 | 0.734 (0.672-0.788) | 0.712 |
| Reciprocal-rank fusion of the three | 0.657 | 0.670 (0.609-0.730) | 0.664 |
| **LambdaRank reranker** | **0.750** | **0.749 (0.687-0.806)** | **0.757** |

Paired difference in NDCG@10 over jobs: reranker beats BM25 by +0.122 (CI +0.074 to +0.166) and TF-IDF by +0.095
(CI +0.053 to +0.138). Against the dense retriever alone the gain is **+0.015 (CI -0.020 to +0.050), not
statistically distinguishable**. Embeddings carry most of the signal; the reranker mainly fixes what lexical
baselines miss.

**Caveats.**
- Evaluation ranks each job's *labelled* candidate list (about 25 per job), not the whole pool, because unlabelled
  pairs have unknown fit. Random scores 0.578 here, so read gains relative to that, not to zero.
- Resume length is among the top reranker features (about 22% of gain), which suggests the labels partly reward
  longer resumes. A real deployment would need audited labels.
- The dataset appears to be synthetic or LLM-generated, not real hiring outcomes. No claim here is about real users.
- Candidates are seen in both train and test (jobs are disjoint), so this measures new-job generalisation, not
  new-candidate generalisation.
- Docker image `Dockerfile.jobmatch` is written but not built (no Docker daemon in the authoring environment).
- Warm-up: first `/match` call loads the embedding model (about 8 s on 1 CPU); later calls took about 0.5 s.

Reproduce: `python scripts/run_jobmatch.py` (about 6 min on 1 CPU), then `uvicorn serve.jobmatch_app:app`.

## Serving latency (measured, local uvicorn, single worker, CPU)

300 requests to `GET /recommend?k=10`: **p50 3.07 ms, p95 3.26 ms**.

## What doesn't work yet (honest list)

- Popularity still wins the headline metrics on this split. The two-tower needs
  sampled-softmax with a proper negative sampler (or full softmax over the 3.7k catalog),
  more epochs, and sequence features to compete on Recall@10.
- No retraining pipeline or online features; artifacts are a one-shot snapshot.
- 48% of test events are cold users/items; the only fallback today is popularity.
  The item tower's genre projection is the hook for a real cold-item path.
- The Dockerfile was written and statically reviewed but **not built** in the authoring
  environment (no Docker daemon there); the app code paths it runs are tested locally.
- The demo deploys nowhere yet: run Streamlit locally, or push to Hugging Face Spaces
  (see below).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python src/data.py 2>/dev/null || python -c "from src import data; data.download()"  # fetch MovieLens 1M
python scripts/run_eda.py            # dataset stats + figures
python scripts/train_all.py prepare  # split (stages can run separately)
python scripts/train_all.py popularity
python scripts/train_all.py mf
python scripts/train_all.py two_tower
python scripts/train_all.py finalize # results/metrics.json
python scripts/cold_start.py         # results/cold_start.json + figure
python scripts/build_index.py        # FAISS index

uvicorn serve.app:app                # API on :8000
python scripts/measure_latency.py    # p50/p95 into results/latency.json
streamlit run app/streamlit_app.py   # demo on :8501
pytest tests/                        # metric, split and API tests
```

Pre-trained artifacts (embeddings, FAISS index, popularity scores) are committed under
`artifacts/`, so the API and demo work straight after `pip install` + downloading the
dataset.

## Docker

```bash
docker build -t rec-engine .
docker run -p 8000:8000 rec-engine
```

## Deploy the demo (Hugging Face Spaces)

1. Create a Space (Streamlit SDK) and push this repo.
2. Add a `packages.txt` if the runner needs system libs (usually not).
3. The committed artifacts make the app start without training.

## Repo layout

```
src/jobmatch/   resume-to-job ranking: data, features, metrics, two-stage matcher
src/            data loading, chronological split, metrics, tracking, models
scripts/        EDA, staged training, cold-start analysis, FAISS build, latency probe
serve/app.py    FastAPI /recommend + /health
app/            Streamlit demo
tests/          pytest: metrics correctness, split integrity, API contract
results/        metrics.json, cold_start.json, latency.json, runs.jsonl, figures
artifacts/      trained embeddings, FAISS index, popularity scores (committed)
docs/           README images
```
