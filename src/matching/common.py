import numpy as np
import pandas as pd

from src.matching.preprocess import clean


def load_data():
    abt = pd.read_csv("data/Abt.csv", encoding="latin-1")
    buy = pd.read_csv("data/Buy.csv", encoding="latin-1")
    gt = pd.read_csv("data/abt_buy_perfectMapping.csv")
    abt["text"] = abt["name"].apply(clean)
    buy["text"] = buy["name"].apply(clean)
    truth = {}
    for a, b in zip(gt["idAbt"], gt["idBuy"]):
        truth.setdefault(a, set()).add(b)
    return abt, buy, truth


def split_abt_ids(abt, truth, seed=42):
    """Abt ürünlerini %60/20/20 train/val/test olarak böler (sadece eşleşmesi olanlar)."""
    ids = np.array([i for i in abt["id"] if i in truth])
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    train = set(ids[: int(n * 0.6)])
    val = set(ids[int(n * 0.6): int(n * 0.8)])
    test = set(ids[int(n * 0.8):])
    return train, val, test


def evaluate(sim, abt_ids, buy_ids, truth, subset=None):
    """sim: (n_abt, n_buy) benzerlik matrisi. subset verilirse sadece o Abt id'leri değerlendirilir."""
    res = {}
    for k in (1, 5, 10):
        hit, total = 0, 0
        for i, aid in enumerate(abt_ids):
            if aid not in truth or (subset is not None and aid not in subset):
                continue
            top = buy_ids[np.argsort(-sim[i])[:k]]
            hit += len(set(top) & truth[aid])
            total += len(truth[aid])
        res[f"R@{k}"] = hit / total
    correct, n = 0, 0
    for i, aid in enumerate(abt_ids):
        if aid not in truth or (subset is not None and aid not in subset):
            continue
        n += 1
        correct += int(buy_ids[sim[i].argmax()] in truth[aid])
    res["P@1"] = correct / n
    return res