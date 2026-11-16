import statistics
import time

import httpx
import pandas as pd

URL = "http://127.0.0.1:8000/match"
titles = pd.read_csv("data/Abt.csv", encoding="latin-1")["name"].dropna().tolist()[:200]

with httpx.Client(timeout=30) as c:
    for t in titles[:20]:  # ısınma, ölçüme dahil değil
        c.post(URL, json={"title": t, "top_k": 5})

    wall, server = [], []
    for t in titles:
        s = time.perf_counter()
        r = c.post(URL, json={"title": t, "top_k": 5})
        wall.append((time.perf_counter() - s) * 1000)
        server.append(r.json()["latency_ms"])


def q(xs, p):
    return statistics.quantiles(xs, n=100)[p - 1]


print(f"İstek sayısı: {len(wall)}")
print(f"Uçtan uca  p50={q(wall,50):.1f} ms  p95={q(wall,95):.1f} ms")
print(f"Sunucu içi p50={q(server,50):.1f} ms  p95={q(server,95):.1f} ms")