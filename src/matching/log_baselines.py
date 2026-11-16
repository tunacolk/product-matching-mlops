import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.matching.common import load_data, split_abt_ids, evaluate

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("abt-buy-product-matching")

abt, buy, truth = load_data()
train_ids, val_ids, test_ids = split_abt_ids(abt, truth)
abt_ids, buy_ids = abt["id"].to_numpy(), buy["id"].to_numpy()

# TF-IDF
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
vec.fit(pd.concat([abt["text"], buy["text"]]))
sim_tfidf = cosine_similarity(vec.transform(abt["text"]), vec.transform(buy["text"]))


def embed_sim(path):
    m = SentenceTransformer(path, device="cuda")
    A = m.encode(abt["text"].tolist(), batch_size=64, normalize_embeddings=True)
    B = m.encode(buy["text"].tolist(), batch_size=64, normalize_embeddings=True)
    return A @ B.T


def log_run(name, params, sim):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_param("eval_split", "val")
        metrics = evaluate(sim, abt_ids, buy_ids, truth, val_ids)
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})


sim_pre = embed_sim("all-MiniLM-L6-v2")
sim_ft = embed_sim("models/minilm-finetuned")

log_run("tfidf", {"method": "tfidf", "analyzer": "char_wb", "ngram": "3-5"}, sim_tfidf)
log_run("minilm-pretrained", {"method": "embedding", "model": "all-MiniLM-L6-v2"}, sim_pre)
log_run("minilm-finetuned", {"method": "embedding", "model": "minilm-finetuned"}, sim_ft)

for alpha in (0.3, 0.5, 0.7):
    sim = alpha * sim_ft + (1 - alpha) * sim_tfidf
    log_run(f"hybrid-a{alpha}", {"method": "hybrid", "alpha": alpha, "model": "minilm-finetuned"}, sim)

print("Bitti. Arayüz için: mlflow ui")