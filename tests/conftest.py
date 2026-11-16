import os

os.environ["SKIP_MODEL_LOAD"] = "1"

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, state


class FakeMatcher:
    """Gerçek modeli taklit eder: sabit, tahmin edilebilir sonuçlar döner."""

    ids = np.arange(100)

    def match(self, title, top_k=5):
        return [
            {
                "id": i,
                "name": f"Fake Product {i}",
                "score": 1.0 - i * 0.1,
                "embedding_score": 0.9,
                "tfidf_score": 0.8,
            }
            for i in range(top_k)
        ]


@pytest.fixture
def client():
    state["matcher"] = FakeMatcher()
    with TestClient(app) as c:
        state["matcher"] = FakeMatcher()  # lifespan state'i temizlemiş olabilir
        yield c
    state.clear()