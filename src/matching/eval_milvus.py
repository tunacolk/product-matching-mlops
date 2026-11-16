import numpy as np

from src.matching.common import load_data, split_abt_ids, evaluate
from src.matching.milvus_matcher import MilvusMatcher

abt, buy, truth = load_data()
train_ids, val_ids, test_ids = split_abt_ids(abt, truth)
abt_ids, buy_ids = abt["id"].to_numpy(), buy["id"].to_numpy()

matcher = MilvusMatcher("models/minilm-finetuned", "data/Buy.csv", "data/Abt.csv")

sim = np.full((len(abt), len(buy)), -1.0)
for i, title in enumerate(abt["name"].fillna("")):
    if abt_ids[i] not in val_ids:
        continue
    for c in matcher.match(title, top_k=matcher.candidate_n):
        sim[i, matcher.id_to_row[c["id"]]] = c["score"]

print("Milvus + hibrit (alpha=0.5, N=50), VAL:")
print({k: round(v, 3) for k, v in evaluate(sim, abt_ids, buy_ids, truth, val_ids).items()})