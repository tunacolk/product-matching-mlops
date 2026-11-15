import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.matching.preprocess import clean

# --- Veri ---
abt = pd.read_csv("data/Abt.csv", encoding="latin-1")
buy = pd.read_csv("data/Buy.csv", encoding="latin-1")
gt = pd.read_csv("data/abt_buy_perfectMapping.csv")

abt["text"] = abt["name"].apply(clean)
buy["text"] = buy["name"].apply(clean)

# Gerçek eşleşmeler: {abt_id: {buy_id, ...}}
truth = {}
for a, b in zip(gt["idAbt"], gt["idBuy"]):
    truth.setdefault(a, set()).add(b)

# --- TF-IDF (karakter n-gram, model kodlarında iyi çalışır) ---
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
vec.fit(pd.concat([abt["text"], buy["text"]]))
A = vec.transform(abt["text"])
B = vec.transform(buy["text"])

sim = cosine_similarity(A, B)  # (1081, 1092)

# --- Değerlendirme: her Abt ürünü için top-k Buy adayı ---
buy_ids = buy["id"].to_numpy()
abt_ids = abt["id"].to_numpy()

for k in (1, 5, 10):
    hit, total = 0, 0
    for i, aid in enumerate(abt_ids):
        if aid not in truth:
            continue
        top = buy_ids[sim[i].argsort()[::-1][:k]]
        hit += len(set(top) & truth[aid])
        total += len(truth[aid])
    print(f"Recall@{k}: {hit / total:.3f}")

# Top-1 precision: ilk aday doğru mu?
correct, n = 0, 0
for i, aid in enumerate(abt_ids):
    if aid not in truth:
        continue
    n += 1
    correct += int(buy_ids[sim[i].argmax()] in truth[aid])
print(f"Precision@1: {correct / n:.3f}")