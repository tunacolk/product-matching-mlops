import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.matching.preprocess import clean

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

abt = pd.read_csv("data/Abt.csv", encoding="latin-1")
buy = pd.read_csv("data/Buy.csv", encoding="latin-1")
gt = pd.read_csv("data/abt_buy_perfectMapping.csv")

abt["text"] = abt["name"].apply(clean)
buy["text"] = buy["name"].apply(clean)

truth = {}
for a, b in zip(gt["idAbt"], gt["idBuy"]):
    truth.setdefault(a, set()).add(b)

# --- Embedding (normalize edilmiş vektörlerde dot product = cosine) ---
model = SentenceTransformer(MODEL_NAME, device="cuda")
A = model.encode(abt["text"].tolist(), batch_size=64,
                 normalize_embeddings=True, show_progress_bar=True)
B = model.encode(buy["text"].tolist(), batch_size=64,
                 normalize_embeddings=True, show_progress_bar=True)

sim = A @ B.T

buy_ids = buy["id"].to_numpy()
abt_ids = abt["id"].to_numpy()

for k in (1, 5, 10):
    hit, total = 0, 0
    for i, aid in enumerate(abt_ids):
        if aid not in truth:
            continue
        top = buy_ids[np.argsort(-sim[i])[:k]]
        hit += len(set(top) & truth[aid])
        total += len(truth[aid])
    print(f"Recall@{k}: {hit / total:.3f}")

correct, n = 0, 0
for i, aid in enumerate(abt_ids):
    if aid not in truth:
        continue
    n += 1
    correct += int(buy_ids[sim[i].argmax()] in truth[aid])
print(f"Precision@1: {correct / n:.3f}")