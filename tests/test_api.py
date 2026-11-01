import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("artifacts/item_index.faiss"),
    reason="requires trained artifacts (run scripts/train_all.py + build_index.py)",
)


def test_health_and_recommend():
    from fastapi.testclient import TestClient
    from serve.app import app
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/recommend", params={"user_id": 1, "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["fallback"] is None
    assert len(body["recommendations"]) == 5
    assert all(rec["title"] for rec in body["recommendations"])
    # unknown user falls back to popularity
    cold = client.get("/recommend", params={"user_id": 999999, "k": 5}).json()
    assert cold["fallback"] == "popularity"
