import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.matching.common import load_data, split_abt_ids, evaluate

abt, buy, truth = load_data()
train_ids, val_ids, test_ids = split_abt_ids(abt, truth)
abt_ids, buy_ids = abt["id"].to_numpy(), buy["id"].to_numpy()

# TF-IDF benzerliği
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
vec.fit(pd.concat([abt["text"], buy["text"]]))
sim_tfidf = cosine_similarity(vec.transform(abt["text"]), vec.transform(buy["text"]))

# Embedding benzerliği
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
A = model.encode(abt["text"].tolist(), batch_size=64, normalize_embeddings=True)
B = model.encode(buy["text"].tolist(), batch_size=64, normalize_embeddings=True)
sim_emb = A @ B.T

# α taraması (sadece VAL üzerinde)
rows = []
for alpha in np.arange(0.0, 1.01, 0.1):
    sim = alpha * sim_emb + (1 - alpha) * sim_tfidf
    rows.append({"alpha": round(alpha, 1), **evaluate(sim, abt_ids, buy_ids, truth, val_ids)})

print(pd.DataFrame(rows).round(3).to_string(index=False))