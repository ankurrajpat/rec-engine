"""Measure /recommend latency against a locally running server.

Usage: start `uvicorn serve.app:app` first, then run this script.
Writes results/latency.json with p50/p95 in milliseconds.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from src import config

BASE = "http://127.0.0.1:8000"
N_REQUESTS = 300
USER_IDS = [1, 25, 100, 500, 1000, 2000, 3000, 4000, 5000, 6040, 999999]


def main():
    lat = []
    with httpx.Client(base_url=BASE, timeout=10) as client:
        client.get("/health").raise_for_status()
        for i in range(N_REQUESTS):
            uid = USER_IDS[i % len(USER_IDS)]
            t0 = time.perf_counter()
            r = client.get("/recommend", params={"user_id": uid, "k": 10})
            r.raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    out = {
        "n_requests": N_REQUESTS,
        "endpoint": "GET /recommend?k=10",
        "p50_ms": round(lat[len(lat) // 2], 2),
        "p95_ms": round(lat[int(len(lat) * 0.95)], 2),
        "mean_ms": round(sum(lat) / len(lat), 2),
        "note": "local uvicorn, single worker, CPU",
    }
    (config.RESULTS_DIR / "latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
