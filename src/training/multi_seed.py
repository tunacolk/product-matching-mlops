import gc
import random

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.matching.common import load_data, split_abt_ids, evaluate

SEEDS = [42, 43, 44]
ALPHA = 0.5
BASE = "all-MiniLM-L6-v2"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("abt-buy-final-3seeds")

abt, buy, truth = load_data()
train_ids, val_ids, test_ids = split_abt_ids(abt, truth)
abt_ids, buy_ids = abt["id"].to_numpy(), buy["id"].to_numpy()
abt_text = dict(zip(abt["id"], abt["text"]))
buy_text = dict(zip(buy["id"], buy["text"]))


def embed_sim(model):
    A = model.encode(abt["text"].tolist(), batch_size=64, normalize_embeddings=True)
    B = model.encode(buy["text"].tolist(), batch_size=64, normalize_embeddings=True)
    return A @ B.T


def both_splits(sim):
    return {"val": evaluate(sim, abt_ids, buy_ids, truth, val_ids),
            "test": evaluate(sim, abt_ids, buy_ids, truth, test_ids)}


def log(run_name, params, res):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for split, metrics in res.items():
            mlflow.log_metrics({f"{split}_{k.replace('@', '_at_')}": v
                                for k, v in metrics.items()})


def train(seed):
    random.seed(seed)
    pairs = [{"anchor": abt_text[a], "positive": buy_text[b]}
             for a in train_ids for b in truth[a]]
    random.shuffle(pairs)
    model = SentenceTransformer(BASE, device="cuda")
    out = f"models/minilm-seed{seed}"
    args = SentenceTransformerTrainingArguments(
        output_dir=out, num_train_epochs=4, per_device_train_batch_size=32,
        learning_rate=2e-5, warmup_steps=0.1, fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no",
        logging_steps=10, report_to="none", seed=seed, data_seed=seed,
    )
    SentenceTransformerTrainer(
        model=model, args=args, train_dataset=Dataset.from_list(pairs),
        loss=MultipleNegativesRankingLoss(model),
    ).train()
    model.save(out)
    return model


# --- Seed'siz yöntemler ---
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
vec.fit(pd.concat([abt["text"], buy["text"]]))
sim_tfidf = cosine_similarity(vec.transform(abt["text"]), vec.transform(buy["text"]))
res_tfidf = both_splits(sim_tfidf)
log("tfidf", {"method": "tfidf"}, res_tfidf)

pre = SentenceTransformer(BASE, device="cuda")
res_pre = both_splits(embed_sim(pre))
log("pretrained", {"method": "embedding", "model": BASE}, res_pre)
del pre

# --- Seed'li yöntemler ---
all_ft, all_hy = [], []
for seed in SEEDS:
    model = train(seed)
    sim_ft = embed_sim(model)
    res_ft = both_splits(sim_ft)
    res_hy = both_splits(ALPHA * sim_ft + (1 - ALPHA) * sim_tfidf)
    log(f"finetuned-seed{seed}", {"method": "finetuned", "seed": seed}, res_ft)
    log(f"hybrid-seed{seed}", {"method": "hybrid", "seed": seed, "alpha": ALPHA}, res_hy)
    all_ft.append(res_ft)
    all_hy.append(res_hy)
    del model, sim_ft
    gc.collect()
    torch.cuda.empty_cache()


def summarize(name, runs, split):
    row = {"yöntem": name}
    for m in ("R@1", "R@5", "R@10", "P@1"):
        v = [r[split][m] for r in runs]
        sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
        row[m] = f"{np.mean(v):.3f} ± {sd:.3f}"
    return row


for split in ("val", "test"):
    rows = [
        summarize("TF-IDF", [res_tfidf], split),
        summarize("Hazır MiniLM", [res_pre], split),
        summarize("Fine-tuned (3 seed)", all_ft, split),
        summarize(f"Hibrit a={ALPHA} (3 seed)", all_hy, split),
    ]
    print(f"\n=== {split.upper()} ===")
    print(pd.DataFrame(rows).to_string(index=False))