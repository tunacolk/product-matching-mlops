import datetime as dt
import logging
import os

import httpx

logger = logging.getLogger("monitoring")
ES_URL = os.getenv("ES_URL")  # tanımlı değilse izleme kapalı (testler, yerel çalışma)
ES_INDEX = os.getenv("ES_INDEX", "matcher-requests")


def build_doc(title, top_k, candidates, latency_ms):
    top1 = candidates[0] if candidates else None
    top2 = candidates[1] if len(candidates) > 1 else None
    return {
        "@timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "query": title,
        "top_k": top_k,
        "latency_ms": latency_ms,
        "backend": os.getenv("MATCHER_BACKEND", "numpy"),
        "top1_id": top1["id"] if top1 else None,
        "top1_score": top1["score"] if top1 else None,
        "top1_embedding_score": top1["embedding_score"] if top1 else None,
        "top1_tfidf_score": top1["tfidf_score"] if top1 else None,
        "score_gap": (top1["score"] - top2["score"]) if top1 and top2 else None,
    }


def log_request(doc):
    """Elasticsearch'e tek kayıt yazar. Hata olursa isteği ASLA bozmaz."""
    if not ES_URL:
        return
    try:
        httpx.post(f"{ES_URL}/{ES_INDEX}/_doc", json=doc, timeout=2.0)
    except Exception as exc:
        logger.warning("ES yazma hatası: %s", exc)