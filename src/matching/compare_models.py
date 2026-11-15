from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from src.matching.common import load_data, split_abt_ids, evaluate

abt, buy, truth = load_data()
train_ids, val_ids, test_ids = split_abt_ids(abt, truth)
abt_ids, buy_ids = abt["id"].to_numpy(), buy["id"].to_numpy()

print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)} Abt ürünü\n")

rows = []

# --- TF-IDF baseline ---
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
vec.fit(pd.concat([abt["text"], buy["text"]]))
sim_tfidf = cosine_similarity(vec.transform(abt["text"]), vec.transform(buy["text"]))
rows.append({"model": "TF-IDF char n-gram", **evaluate(sim_tfidf, abt_ids, buy_ids, truth, val_ids)})

# --- Hazır embedding modelleri ---
for name in ["paraphrase-multilingual-MiniLM-L12-v2",
             "all-MiniLM-L6-v2",
             "all-mpnet-base-v2"]:
    model = SentenceTransformer(name, device="cuda")
    A = model.encode(abt["text"].tolist(), batch_size=64, normalize_embeddings=True)
    B = model.encode(buy["text"].tolist(), batch_size=64, normalize_embeddings=True)
    rows.append({"model": name, **evaluate(A @ B.T, abt_ids, buy_ids, truth, val_ids)})
    del model

print(pd.DataFrame(rows).round(3).to_string(index=False))