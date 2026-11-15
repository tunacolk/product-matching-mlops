import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.matching.common import load_data, split_abt_ids, evaluate

abt, buy, truth = load_data()
train_ids, val_ids, test_ids = split_abt_ids(abt, truth)
abt_ids, buy_ids = abt["id"].to_numpy(), buy["id"].to_numpy()

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
vec.fit(pd.concat([abt["text"], buy["text"]]))
sim_tfidf = cosine_similarity(vec.transform(abt["text"]), vec.transform(buy["text"]))

model = SentenceTransformer("models/minilm-finetuned", device="cuda")
A = model.encode(abt["text"].tolist(), batch_size=64, normalize_embeddings=True)
B = model.encode(buy["text"].tolist(), batch_size=64, normalize_embeddings=True)
sim_ft = A @ B.T

rows = [{"yöntem": "TF-IDF", **evaluate(sim_tfidf, abt_ids, buy_ids, truth, val_ids)},
        {"yöntem": "Fine-tuned", **evaluate(sim_ft, abt_ids, buy_ids, truth, val_ids)}]
for alpha in (0.3, 0.5, 0.7):
    sim = alpha * sim_ft + (1 - alpha) * sim_tfidf
    rows.append({"yöntem": f"Hibrit a={alpha}", **evaluate(sim, abt_ids, buy_ids, truth, val_ids)})

print(pd.DataFrame(rows).round(3).to_string(index=False))