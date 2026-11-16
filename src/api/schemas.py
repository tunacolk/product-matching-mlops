from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    title: str = Field(..., min_length=1, examples=["Sony Turntable PSLX350H"])
    top_k: int = Field(5, ge=1, le=50)


class Candidate(BaseModel):
    id: int
    name: str
    score: float
    embedding_score: float
    tfidf_score: float


class MatchResponse(BaseModel):
    query: str
    candidates: list[Candidate]
    latency_ms: float