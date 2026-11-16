import logging
import time
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from src.api.schemas import MatchRequest, MatchResponse
from src.matching.matcher import HybridMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matcher-api")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        logger.info("Model ve katalog yükleniyor...")
        state["matcher"] = HybridMatcher(
            model_path="models/minilm-finetuned",
            catalog_csv="data/Buy.csv",
            abt_csv="data/Abt.csv",
            alpha=0.5,
        )
        logger.info("Hazır. Katalog boyutu: %d", len(state["matcher"].ids))
    yield
    state.clear()


app = FastAPI(title="Product Matching API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "catalog_size": len(state["matcher"].ids)}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    start = time.perf_counter()
    candidates = state["matcher"].match(req.title, req.top_k)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("match title=%r top1=%s latency_ms=%.1f",
                req.title, candidates[0]["score"] if candidates else None, latency_ms)
    return MatchResponse(query=req.title, candidates=candidates, latency_ms=latency_ms)