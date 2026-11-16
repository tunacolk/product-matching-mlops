def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["catalog_size"] == 100


def test_match_returns_top_k(client):
    r = client.post("/match", json={"title": "Sony Turntable", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "Sony Turntable"
    assert len(body["candidates"]) == 3
    assert body["latency_ms"] >= 0


def test_match_scores_descending(client):
    r = client.post("/match", json={"title": "x", "top_k": 5})
    scores = [c["score"] for c in r.json()["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_match_default_top_k(client):
    r = client.post("/match", json={"title": "Sony"})
    assert len(r.json()["candidates"]) == 5


def test_empty_title_rejected(client):
    r = client.post("/match", json={"title": ""})
    assert r.status_code == 422


def test_top_k_out_of_range_rejected(client):
    assert client.post("/match", json={"title": "x", "top_k": 0}).status_code == 422
    assert client.post("/match", json={"title": "x", "top_k": 51}).status_code == 422


def test_missing_title_rejected(client):
    assert client.post("/match", json={}).status_code == 422