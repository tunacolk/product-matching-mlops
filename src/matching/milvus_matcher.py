import numpy as np
import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.matching.preprocess import clean


class MilvusMatcher:
    """Milvus ile aday getirir (embedding), TF-IDF ile hibrit yeniden sıralar."""

    def __init__(self, model_path, catalog_csv, abt_csv,
                 uri="http://localhost:19530", collection="buy_products",
                 alpha=0.5, candidate_n=50):
        self.alpha = alpha
        self.candidate_n = candidate_n
        self.collection = collection
        self.model = SentenceTransformer(model_path)
        self.client = MilvusClient(uri=uri)

        buy = pd.read_csv(catalog_csv, encoding="latin-1")
        abt = pd.read_csv(abt_csv, encoding="latin-1")
        self.ids = buy["id"].to_numpy()
        self.names = buy["name"].fillna("").tolist()
        self.id_to_row = {int(i): r for r, i in enumerate(self.ids)}

        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.vec.fit(pd.concat([abt["name"].apply(clean), buy["name"].apply(clean)]))
        self.tfidf_matrix = self.vec.transform(buy["name"].apply(clean))

    def match(self, title, top_k=5):
        q = clean(title)
        q_emb = self.model.encode([q], normalize_embeddings=True)

        res = self.client.search(
            self.collection,
            data=q_emb.tolist(),
            limit=self.candidate_n,
            search_params={"params": {"ef": 128}},
        )[0]
        cand_ids = [int(h["id"]) for h in res]
        emb_scores = np.array([h["distance"] for h in res])

        rows = [self.id_to_row[i] for i in cand_ids]
        q_tfidf = self.vec.transform([q])
        tfidf_scores = (self.tfidf_matrix[rows] @ q_tfidf.T).toarray().ravel()

        score = self.alpha * emb_scores + (1 - self.alpha) * tfidf_scores
        order = np.argsort(-score)[:top_k]
        return [
            {
                "id": cand_ids[j],
                "name": self.names[rows[j]],
                "score": float(score[j]),
                "embedding_score": float(emb_scores[j]),
                "tfidf_score": float(tfidf_scores[j]),
            }
            for j in order
        ]