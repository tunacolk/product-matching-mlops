from src.api.monitoring import build_doc, log_request


def test_build_doc_fields():
    cands = [
        {"id": 1, "name": "a", "score": 0.9, "embedding_score": 0.8, "tfidf_score": 1.0},
        {"id": 2, "name": "b", "score": 0.5, "embedding_score": 0.5, "tfidf_score": 0.5},
    ]
    doc = build_doc("sony", 2, cands, 12.5)
    assert doc["top1_id"] == 1
    assert abs(doc["score_gap"] - 0.4) < 1e-9
    assert doc["latency_ms"] == 12.5


def test_build_doc_empty_candidates():
    doc = build_doc("x", 1, [], 1.0)
    assert doc["top1_id"] is None and doc["score_gap"] is None


def test_log_request_noop_without_es():
    log_request({"a": 1})  # ES_URL tanımsız: hata vermemeli